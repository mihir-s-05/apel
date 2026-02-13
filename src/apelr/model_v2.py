from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class APELRV2ModelConfig:
    vocab_size: int
    num_plan_states: int = 8
    num_experts: int = 8
    chunk_size: int = 16
    token_dim: int = 256
    hidden_dim: int = 768
    num_layers: int = 2
    dropout: float = 0.1
    planner_self_bias: float = 1.8
    planner_context_scale: float = 1.0
    future_horizon_chunks: int = 2
    planner_temperature: float = 1.0
    token_filtering: bool = True
    lookahead_horizon: int = 2
    lookahead_feedback_scale: float = 0.25
    async_planner: bool = True
    eps: float = 1e-9


class APELRV2Model(nn.Module):
    """
    V2 planner-required architecture.

    Core difference from V1:
    - Token logits are produced by planner-conditioned expert heads only.
      There is no high-capacity shared bypass head.
    """

    def __init__(self, cfg: APELRV2ModelConfig) -> None:
        super().__init__()
        if cfg.num_experts != cfg.num_plan_states:
            raise ValueError("For v2, num_experts must equal num_plan_states.")
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.num_plan_states = cfg.num_plan_states
        self.num_experts = cfg.num_experts
        self.chunk_size = cfg.chunk_size
        self.eps = cfg.eps

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.token_dim)
        self.backbone = nn.GRU(
            input_size=cfg.token_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)

        self.planner_init_logits = nn.Parameter(torch.zeros(cfg.num_plan_states))
        self.planner_transition_logits = nn.Parameter(torch.zeros(cfg.num_plan_states, cfg.num_plan_states))
        self.planner_obs_proj = nn.Linear(cfg.hidden_dim, cfg.num_plan_states)
        self.planner_boundary_proj = nn.Linear(cfg.hidden_dim, cfg.num_plan_states)

        self.expert_heads = nn.ModuleList(
            [nn.Linear(cfg.hidden_dim, cfg.vocab_size) for _ in range(cfg.num_experts)]
        )

        self.plan_state_emb = nn.Embedding(cfg.num_plan_states, cfg.hidden_dim)
        self.future_chunk_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        for head in self.expert_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.xavier_uniform_(self.planner_obs_proj.weight)
        nn.init.zeros_(self.planner_obs_proj.bias)
        nn.init.xavier_uniform_(self.planner_boundary_proj.weight)
        nn.init.zeros_(self.planner_boundary_proj.bias)
        nn.init.normal_(self.plan_state_emb.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.future_chunk_proj.weight)
        nn.init.zeros_(self.planner_init_logits)
        nn.init.zeros_(self.planner_transition_logits)
        with torch.no_grad():
            idx = torch.arange(self.num_plan_states, device=self.planner_transition_logits.device)
            self.planner_transition_logits[idx, idx] = self.cfg.planner_self_bias

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def transition_matrix(self) -> torch.Tensor:
        return F.softmax(self.planner_transition_logits, dim=-1)

    def _belief_log(self, belief: torch.Tensor) -> torch.Tensor:
        return (belief + self.eps).log()

    def _initial_belief(self, batch_size: int, device: torch.device) -> torch.Tensor:
        pi0 = F.softmax(self.planner_init_logits, dim=-1)
        return pi0.unsqueeze(0).expand(batch_size, -1).to(device)

    def _chunk_summaries(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, hdim = h.shape
        num_chunks = math.ceil(seq_len / self.chunk_size)
        chunks: list[torch.Tensor] = []
        for c in range(num_chunks):
            s = c * self.chunk_size
            e = min(seq_len, s + self.chunk_size)
            chunks.append(h[:, s:e, :].mean(dim=1))
        chunk_summary = torch.stack(chunks, dim=1)
        token_chunk_idx = torch.arange(seq_len, device=h.device) // self.chunk_size
        return chunk_summary, token_chunk_idx

    def _planner_states(
        self,
        chunk_summary: torch.Tensor,
        *,
        planner_mode: str = "normal",
        forced_state: int | None = None,
        commitment: str = "soft",
        planner_temperature: float = 1.0,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
        - states used for token gating [B, C, K]
        - posterior beliefs [B, C, K] before optional hard commitment
        """
        bsz, num_chunks, _ = chunk_summary.shape
        device = chunk_summary.device

        trans = self.transition_matrix()
        uniform = torch.full((bsz, self.num_plan_states), 1.0 / self.num_plan_states, device=device)

        posteriors: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        belief = self._initial_belief(bsz, device)

        for c in range(num_chunks):
            if c > 0:
                prior = belief @ trans
                prev_summary = chunk_summary[:, c - 1, :]
                ctx_bias = self.cfg.planner_context_scale * self.planner_boundary_proj(prev_summary)
                obs_bias = self.planner_obs_proj(prev_summary)
                belief = F.softmax(self._belief_log(prior) + ctx_bias + obs_bias, dim=-1)

            posterior = belief
            posteriors.append(posterior)

            if planner_mode == "uniform_mask":
                state = uniform
            elif planner_mode == "forced_state":
                if forced_state is None:
                    raise ValueError("forced_state must be set when planner_mode='forced_state'.")
                if forced_state < 0 or forced_state >= self.num_plan_states:
                    raise ValueError(f"forced_state={forced_state} out of range.")
                state = torch.zeros_like(posterior)
                state[:, forced_state] = 1.0
            else:
                if commitment == "gumbel_st" and training:
                    tau = max(float(planner_temperature), 1e-4)
                    state = F.gumbel_softmax((posterior + self.eps).log(), tau=tau, hard=True, dim=-1)
                else:
                    state = posterior

            states.append(state)

        return torch.stack(states, dim=1), torch.stack(posteriors, dim=1)

    def _repetition_unlikelihood_loss(
        self,
        mix_log_probs: torch.Tensor,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        window: int,
    ) -> torch.Tensor:
        if window <= 0:
            return torch.zeros((), device=mix_log_probs.device)
        probs = mix_log_probs.exp()
        bsz, seq_len = input_ids.shape
        losses: list[torch.Tensor] = []
        for b in range(bsz):
            for t in range(seq_len):
                s = max(0, t - window)
                if s == t:
                    continue
                cand = torch.unique(input_ids[b, s:t])
                tgt = target_ids[b, t]
                cand = cand[cand != tgt]
                if cand.numel() == 0:
                    continue
                p = probs[b, t, cand].clamp(min=0.0, max=1.0 - 1e-6)
                losses.append(-torch.log1p(-p).mean())
        if not losses:
            return torch.zeros((), device=mix_log_probs.device)
        return torch.stack(losses).mean()

    def _expert_log_probs(self, h: torch.Tensor) -> torch.Tensor:
        logits_k = torch.stack([head(self.dropout(h)) for head in self.expert_heads], dim=2)
        return F.log_softmax(logits_k, dim=-1)

    def _mixture_log_probs(
        self,
        expert_log_probs: torch.Tensor,
        chunk_states: torch.Tensor,
        token_chunk_idx: torch.Tensor,
    ) -> torch.Tensor:
        token_states = chunk_states[:, token_chunk_idx, :]
        return torch.logsumexp(self._belief_log(token_states).unsqueeze(-1) + expert_log_probs, dim=2)

    def _lookahead_first_mean(
        self,
        belief: torch.Tensor,
        transition: torch.Tensor,
        steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if steps <= 0:
            return belief, belief
        cur = belief
        first: torch.Tensor | None = None
        acc = torch.zeros_like(belief)
        for s in range(steps):
            cur = cur @ transition
            if s == 0:
                first = cur
            acc = acc + cur
        mean = acc / float(steps)
        return (belief if first is None else first), mean

    def _planner_gate_with_lookahead(
        self,
        belief: torch.Tensor,
        transition: torch.Tensor,
        *,
        planner_temperature: float,
        lookahead_steps: int,
        feedback_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mixed = belief
        first = belief
        if lookahead_steps > 0 and feedback_scale > 0.0:
            first, future_mean = self._lookahead_first_mean(belief, transition, lookahead_steps)
            alpha = min(max(float(feedback_scale), 0.0), 1.0)
            mixed = (1.0 - alpha) * belief + alpha * future_mean
        tau = max(float(planner_temperature), 1e-4)
        gate = F.softmax(self._belief_log(mixed) / tau, dim=-1)
        return gate, first

    def _chunk_states_with_feedback(
        self,
        chunk_states: torch.Tensor,
        transition: torch.Tensor,
        *,
        planner_mode: str,
        planner_temperature: float,
        lookahead_steps: int,
        feedback_scale: float,
    ) -> torch.Tensor:
        if chunk_states.shape[1] == 0:
            return chunk_states
        if planner_mode != "normal":
            tau = max(float(planner_temperature), 1e-4)
            return F.softmax(self._belief_log(chunk_states) / tau, dim=-1)
        gates: list[torch.Tensor] = []
        for c in range(chunk_states.shape[1]):
            gate_c, _ = self._planner_gate_with_lookahead(
                chunk_states[:, c, :],
                transition,
                planner_temperature=planner_temperature,
                lookahead_steps=int(lookahead_steps),
                feedback_scale=float(feedback_scale),
            )
            gates.append(gate_c)
        return torch.stack(gates, dim=1)

    def _future_contrastive_loss(self, chunk_states: torch.Tensor, chunk_summary: torch.Tensor) -> torch.Tensor:
        bsz, num_chunks, _ = chunk_summary.shape
        if num_chunks <= 1:
            return torch.zeros((), device=chunk_summary.device)
        horizon = min(self.cfg.future_horizon_chunks, num_chunks - 1)
        if horizon <= 0:
            return torch.zeros((), device=chunk_summary.device)

        plan_vec = chunk_states @ self.plan_state_emb.weight
        plan_vec = F.normalize(plan_vec, dim=-1)
        future_vec = F.normalize(self.future_chunk_proj(chunk_summary), dim=-1)

        losses: list[torch.Tensor] = []
        for d in range(1, horizon + 1):
            anchor = plan_vec[:, :-d, :].reshape(-1, plan_vec.shape[-1])
            positive = future_vec[:, d:, :].reshape(-1, future_vec.shape[-1])
            if anchor.numel() == 0:
                continue
            logits = anchor @ positive.t()
            logits = logits / max(self.cfg.planner_temperature, 1e-4)
            labels = torch.arange(anchor.shape[0], device=anchor.device)
            losses.append(F.cross_entropy(logits, labels))
        if not losses:
            return torch.zeros((), device=chunk_summary.device)
        return torch.stack(losses).mean()

    def _pairwise_js_div_loss(self, expert_log_probs: torch.Tensor) -> torch.Tensor:
        probs = expert_log_probs.exp().reshape(-1, self.num_experts, self.vocab_size)
        logs = expert_log_probs.reshape(-1, self.num_experts, self.vocab_size)
        num_pairs = 0
        js_sum = torch.zeros((), device=expert_log_probs.device)
        for i in range(self.num_experts):
            pi = probs[:, i, :]
            lpi = logs[:, i, :]
            for j in range(i + 1, self.num_experts):
                pj = probs[:, j, :]
                lpj = logs[:, j, :]
                m = 0.5 * (pi + pj)
                lm = (m + self.eps).log()
                kl_im = torch.sum(pi * (lpi - lm), dim=-1)
                kl_jm = torch.sum(pj * (lpj - lm), dim=-1)
                js = 0.5 * (kl_im + kl_jm)
                js_sum = js_sum + js.mean()
                num_pairs += 1
        if num_pairs == 0:
            return torch.zeros((), device=expert_log_probs.device)
        return js_sum / num_pairs

    def compute_losses(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        planner_mode: str = "normal",
        forced_state: int | None = None,
        commitment: str = "soft",
        planner_temperature: float | None = None,
        lookahead_steps: int | None = None,
        lookahead_feedback_scale: float | None = None,
        token_filtering: bool | None = None,
        rep_unlikelihood_window: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Returns base NLL and aux terms for V2 training/eval.
        """
        x = self.token_emb(input_ids)
        h, _ = self.backbone(x)
        chunk_summary, token_chunk_idx = self._chunk_summaries(h)
        chunk_states, chunk_post = self._planner_states(
            chunk_summary,
            planner_mode=planner_mode,
            forced_state=forced_state,
            commitment=commitment,
            planner_temperature=float(self.cfg.planner_temperature if planner_temperature is None else planner_temperature),
            training=self.training,
        )

        expert_log_probs = self._expert_log_probs(h)
        trans = self.transition_matrix()
        effective_token_filtering = bool(self.cfg.token_filtering if token_filtering is None else token_filtering)
        effective_lookahead_steps = max(
            0,
            int(self.cfg.lookahead_horizon if lookahead_steps is None else lookahead_steps),
        )
        effective_feedback_scale = min(
            max(float(self.cfg.lookahead_feedback_scale if lookahead_feedback_scale is None else lookahead_feedback_scale), 0.0),
            1.0,
        )
        if effective_token_filtering:
            bsz, seq_len = input_ids.shape
            mix_steps: list[torch.Tensor] = []
            cur_chunk = int(token_chunk_idx[0].item()) if seq_len > 0 else 0
            belief_t = chunk_states[:, cur_chunk, :]
            for t in range(seq_len):
                c = int(token_chunk_idx[t].item())
                if c != cur_chunk:
                    cur_chunk = c
                    belief_t = chunk_states[:, cur_chunk, :]
                gate_t, _ = self._planner_gate_with_lookahead(
                    belief_t,
                    trans,
                    planner_temperature=planner_temperature if planner_temperature is not None else self.cfg.planner_temperature,
                    lookahead_steps=effective_lookahead_steps if planner_mode == "normal" else 0,
                    feedback_scale=effective_feedback_scale if planner_mode == "normal" else 0.0,
                )
                mix_t = torch.logsumexp(self._belief_log(gate_t).unsqueeze(-1) + expert_log_probs[:, t, :, :], dim=1)
                mix_steps.append(mix_t)
                tgt = target_ids[:, t]
                obs_logp = torch.gather(
                    expert_log_probs[:, t, :, :],
                    dim=-1,
                    index=tgt.view(-1, 1, 1).expand(-1, self.num_plan_states, 1),
                ).squeeze(-1)
                if planner_mode == "normal":
                    belief_t = F.softmax(self._belief_log(belief_t) + obs_logp, dim=-1)
            mix_log_probs = torch.stack(mix_steps, dim=1)
        else:
            chunk_gates = self._chunk_states_with_feedback(
                chunk_states,
                trans,
                planner_mode=planner_mode,
                planner_temperature=planner_temperature if planner_temperature is not None else self.cfg.planner_temperature,
                lookahead_steps=effective_lookahead_steps,
                feedback_scale=effective_feedback_scale,
            )
            mix_log_probs = self._mixture_log_probs(expert_log_probs, chunk_gates, token_chunk_idx)
        tok_nll = F.nll_loss(
            mix_log_probs.reshape(-1, self.vocab_size),
            target_ids.reshape(-1),
            reduction="mean",
        )

        boundary_entropy = -(chunk_post * (chunk_post + self.eps).log()).sum(dim=-1).mean()
        usage = chunk_states.mean(dim=(0, 1))
        uniform = torch.full_like(usage, 1.0 / self.num_plan_states)
        usage_kl = torch.sum(usage * ((usage + self.eps).log() - uniform.log()))

        future_contrastive = self._future_contrastive_loss(chunk_states, chunk_summary)
        plan_js_div = self._pairwise_js_div_loss(expert_log_probs)
        rep_unlikelihood = self._repetition_unlikelihood_loss(
            mix_log_probs,
            input_ids=input_ids,
            target_ids=target_ids,
            window=rep_unlikelihood_window,
        )

        state_idx = torch.argmax(chunk_states, dim=-1)
        if state_idx.shape[1] > 1:
            state_persistence = (state_idx[:, 1:] == state_idx[:, :-1]).float().mean()
        else:
            state_persistence = torch.tensor(1.0, device=state_idx.device)
        expert_utilization = -(usage * (usage + self.eps).log()).sum() / math.log(self.num_plan_states)

        aux = {
            "boundary_entropy": boundary_entropy,
            "usage_kl_to_uniform": usage_kl,
            "future_contrastive_loss": future_contrastive,
            "plan_js_div_loss": plan_js_div,
            "rep_unlikelihood_loss": rep_unlikelihood,
            "state_persistence": state_persistence,
            "expert_utilization": expert_utilization,
        }
        return tok_nll, aux

    @torch.inference_mode()
    def planner_usage_metrics(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        commitment: str = "soft",
        planner_temperature: float | None = None,
    ) -> dict[str, torch.Tensor]:
        base_nll, aux = self.compute_losses(
            input_ids,
            target_ids,
            planner_mode="normal",
            commitment=commitment,
            planner_temperature=planner_temperature,
        )
        masked_nll, _ = self.compute_losses(
            input_ids,
            target_ids,
            planner_mode="uniform_mask",
            commitment=commitment,
            planner_temperature=planner_temperature,
        )
        planner_mask_delta = masked_nll - base_nll

        bsz, seq_len = input_ids.shape
        x = self.token_emb(input_ids)
        h, _ = self.backbone(x)
        chunk_summary, token_chunk_idx = self._chunk_summaries(h)
        expert_log_probs = self._expert_log_probs(h)

        s0, _ = self._planner_states(chunk_summary, planner_mode="forced_state", forced_state=0, training=False)
        s1, _ = self._planner_states(
            chunk_summary,
            planner_mode="forced_state",
            forced_state=min(1, self.num_plan_states - 1),
            training=False,
        )
        log_mix0 = self._mixture_log_probs(expert_log_probs, s0, token_chunk_idx)
        log_mix1 = self._mixture_log_probs(expert_log_probs, s1, token_chunk_idx)
        p0 = log_mix0.exp()
        p1 = log_mix1.exp()
        m = 0.5 * (p0 + p1)
        log_m = (m + self.eps).log()
        js01 = 0.5 * (
            torch.sum(p0 * (log_mix0 - log_m), dim=-1).mean()
            + torch.sum(p1 * (log_mix1 - log_m), dim=-1).mean()
        )

        out = dict(aux)
        out["planner_mask_delta_loss"] = planner_mask_delta
        out["forced_state_divergence"] = js01
        out["nll"] = base_nll
        return out

    @torch.inference_mode()
    def generate_planned(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        eos_id: int | None = None,
        lookahead_steps: int = 0,
        force_state: int | None = None,
        freeze_planner: bool = False,
        planner_temperature: float = 1.0,
        lookahead_feedback_scale: float | None = None,
        async_planner: bool | None = None,
        token_filtering: bool | None = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> tuple[list[int], list[list[float]]]:
        self.eval()
        device = next(self.parameters()).device
        if len(prompt_ids) == 0:
            raise ValueError("prompt_ids cannot be empty")

        seq = list(prompt_ids)
        hidden: torch.Tensor | None = None

        belief = self._initial_belief(1, device)
        trans = self.transition_matrix()
        lookahead_trace: list[list[float]] = []
        think_steps = max(0, int(lookahead_steps))
        feedback_scale = float(
            self.cfg.lookahead_feedback_scale if lookahead_feedback_scale is None else lookahead_feedback_scale
        )
        feedback_scale = min(max(feedback_scale, 0.0), 1.0)
        use_token_filtering = bool(self.cfg.token_filtering if token_filtering is None else token_filtering)
        use_async_planner = bool(self.cfg.async_planner if async_planner is None else async_planner)
        token_pos = 0
        chunk_hidden_buffer: list[torch.Tensor] = []
        planner_pool: ThreadPoolExecutor | None = None
        planner_future: Future[tuple[int, torch.Tensor, torch.Tensor]] | None = None
        latest_first_cpu: torch.Tensor | None = None
        latest_mean_cpu: torch.Tensor | None = None
        latest_forecast_token_pos: int | None = None
        trans_cpu = trans.detach().to("cpu")

        def maybe_boundary_update() -> None:
            nonlocal belief
            if freeze_planner:
                return
            if len(chunk_hidden_buffer) == 0:
                return
            chunk_mean = torch.stack(chunk_hidden_buffer, dim=1).mean(dim=1)
            prior = belief @ trans
            ctx = self.cfg.planner_context_scale * self.planner_boundary_proj(chunk_mean)
            obs = self.planner_obs_proj(chunk_mean)
            belief = F.softmax(self._belief_log(prior) + ctx + obs, dim=-1)

        def banned_tokens_for_no_repeat_ngram(cur_seq: list[int], n: int) -> set[int]:
            if n <= 1:
                return set()
            if len(cur_seq) < n - 1:
                return set()
            prefix = tuple(cur_seq[-(n - 1) :])
            banned: set[int] = set()
            for i in range(0, len(cur_seq) - n + 1):
                ngram = tuple(cur_seq[i : i + n])
                if ngram[:-1] == prefix:
                    banned.add(int(ngram[-1]))
            return banned

        def planner_source_belief() -> torch.Tensor:
            if force_state is None:
                return belief
            src = torch.zeros_like(belief)
            src[:, int(force_state)] = 1.0
            return src

        def compute_lookahead_cpu(
            forecast_token_pos: int,
            belief_cpu: torch.Tensor,
        ) -> tuple[int, torch.Tensor, torch.Tensor]:
            if think_steps <= 0:
                return forecast_token_pos, belief_cpu, belief_cpu
            cur = belief_cpu
            first: torch.Tensor | None = None
            acc = torch.zeros_like(belief_cpu)
            for s in range(think_steps):
                cur = cur @ trans_cpu
                if s == 0:
                    first = cur
                acc = acc + cur
            mean = acc / float(think_steps)
            return forecast_token_pos, (belief_cpu if first is None else first), mean

        def maybe_collect_async_forecast() -> None:
            nonlocal planner_future, latest_first_cpu, latest_mean_cpu, latest_forecast_token_pos
            if planner_future is None or not planner_future.done():
                return
            forecast_token_pos, first, mean = planner_future.result()
            latest_forecast_token_pos = int(forecast_token_pos)
            latest_first_cpu = first
            latest_mean_cpu = mean
            planner_future = None

        def maybe_submit_async_forecast(src_belief: torch.Tensor, forecast_token_pos: int) -> None:
            nonlocal planner_future
            if planner_pool is None or think_steps <= 0:
                return
            if planner_future is None:
                planner_future = planner_pool.submit(
                    compute_lookahead_cpu,
                    int(forecast_token_pos),
                    src_belief.detach().to("cpu"),
                )

        if use_async_planner and think_steps > 0:
            planner_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="apel_planner")

        for tok in prompt_ids:
            tok_t = torch.tensor([[tok]], device=device, dtype=torch.long)
            out, hidden = self.backbone(self.token_emb(tok_t), hidden)
            chunk_hidden_buffer.append(out[:, -1, :])
            token_pos += 1
            if token_pos % self.chunk_size == 0:
                maybe_boundary_update()
                chunk_hidden_buffer = []

        maybe_submit_async_forecast(planner_source_belief(), token_pos)
        try:
            for _ in range(max_new_tokens):
                maybe_collect_async_forecast()

                state = hidden[-1, :, :] if hidden is not None else torch.zeros((1, self.cfg.hidden_dim), device=device)
                expert_logits = torch.stack([head(state) for head in self.expert_heads], dim=1)
                expert_log_probs = F.log_softmax(expert_logits, dim=-1).squeeze(0)

                source_belief = planner_source_belief()
                first_forecast = source_belief
                future_mean = source_belief
                if think_steps > 0:
                    if (
                        latest_first_cpu is not None
                        and latest_mean_cpu is not None
                        and latest_forecast_token_pos == token_pos
                    ):
                        first_forecast = latest_first_cpu.to(device=device)
                        future_mean = latest_mean_cpu.to(device=device)
                    else:
                        first_forecast, future_mean = self._lookahead_first_mean(source_belief, trans, think_steps)

                if force_state is not None:
                    mixed_gate = source_belief
                elif think_steps > 0 and feedback_scale > 0.0:
                    mixed_gate = (1.0 - feedback_scale) * source_belief + feedback_scale * future_mean
                else:
                    mixed_gate = source_belief
                gate = F.softmax(self._belief_log(mixed_gate.squeeze(0)) / max(planner_temperature, 1e-4), dim=-1)

                mix_log = torch.logsumexp((gate + self.eps).log().unsqueeze(-1) + expert_log_probs, dim=0)
                if repetition_penalty > 1.0:
                    seen = torch.tensor(sorted(set(seq)), device=device, dtype=torch.long)
                    mix_log = mix_log.clone()
                    mix_log[seen] = mix_log[seen] - math.log(repetition_penalty)
                if temperature != 1.0:
                    probs = F.softmax(mix_log / max(temperature, 1e-4), dim=-1)
                else:
                    probs = F.softmax(mix_log, dim=-1)

                if no_repeat_ngram_size > 0:
                    banned = banned_tokens_for_no_repeat_ngram(seq, int(no_repeat_ngram_size))
                    if banned:
                        banned_idx = torch.tensor(sorted(banned), device=device, dtype=torch.long)
                        probs = probs.clone()
                        probs[banned_idx] = 0.0

                if top_k is not None and top_k > 0 and top_k < probs.numel():
                    top_vals, top_idx = torch.topk(probs, k=top_k)
                    probs = torch.zeros_like(probs).scatter(0, top_idx, top_vals)

                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=0)
                    keep = cum <= top_p
                    if keep.numel() > 0:
                        keep[0] = True
                    filtered = torch.zeros_like(probs)
                    filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
                    probs = filtered

                probs_sum = probs.sum()
                if probs_sum <= 0:
                    probs = F.softmax(mix_log, dim=-1)
                else:
                    probs = probs / probs_sum
                next_id = int(torch.multinomial(probs, 1).item())

                seq.append(next_id)
                if use_token_filtering and (not freeze_planner) and force_state is None:
                    obs_logp = expert_log_probs[:, next_id].unsqueeze(0)
                    belief = F.softmax(self._belief_log(belief) + obs_logp, dim=-1)
                tok_t = torch.tensor([[next_id]], device=device, dtype=torch.long)
                out, hidden = self.backbone(self.token_emb(tok_t), hidden)
                chunk_hidden_buffer.append(out[:, -1, :])
                token_pos += 1

                if think_steps > 0:
                    lookahead_trace.append(first_forecast.squeeze(0).detach().cpu().tolist())

                if token_pos % self.chunk_size == 0:
                    maybe_boundary_update()
                    chunk_hidden_buffer = []

                maybe_submit_async_forecast(planner_source_belief(), token_pos)

                if eos_id is not None and next_id == eos_id:
                    break
        finally:
            if planner_pool is not None:
                planner_pool.shutdown(wait=False, cancel_futures=True)

        return seq, lookahead_trace
