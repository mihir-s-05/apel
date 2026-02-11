from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class APELRModelConfig:
    vocab_size: int
    num_plan_states: int = 8
    chunk_size: int = 16
    token_dim: int = 256
    hidden_dim: int = 768
    fusion_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    planner_context_scale: float = 1.0
    planner_self_bias: float = 1.5
    eps: float = 1e-9


class APELRModel(nn.Module):
    def __init__(self, cfg: APELRModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.num_plan_states = cfg.num_plan_states
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
        self.plan_emb = nn.Embedding(cfg.num_plan_states, cfg.fusion_dim)
        self.state_proj = nn.Linear(cfg.hidden_dim, cfg.fusion_dim)
        self.plan_proj = nn.Linear(cfg.fusion_dim, cfg.fusion_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.out_proj = nn.Linear(cfg.fusion_dim, cfg.vocab_size)
        self.plan_vocab_bias = nn.Parameter(torch.zeros(cfg.num_plan_states, cfg.vocab_size))
        self.planner_ctx_proj = nn.Linear(cfg.hidden_dim, cfg.num_plan_states)

        self.planner_init_logits = nn.Parameter(torch.zeros(cfg.num_plan_states))
        self.planner_transition_logits = nn.Parameter(torch.zeros(cfg.num_plan_states, cfg.num_plan_states))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.plan_emb.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.state_proj.weight)
        nn.init.zeros_(self.state_proj.bias)
        nn.init.xavier_uniform_(self.plan_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.plan_vocab_bias)
        nn.init.xavier_uniform_(self.planner_ctx_proj.weight)
        nn.init.zeros_(self.planner_ctx_proj.bias)
        nn.init.zeros_(self.planner_init_logits)
        nn.init.zeros_(self.planner_transition_logits)
        with torch.no_grad():
            idx = torch.arange(self.num_plan_states, device=self.planner_transition_logits.device)
            self.planner_transition_logits[idx, idx] = self.cfg.planner_self_bias

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def transition_matrix(self) -> torch.Tensor:
        # Row-stochastic matrix P[i, j] = p(z_{c+1}=j | z_c=i)
        return F.softmax(self.planner_transition_logits, dim=-1)

    def initial_belief(self, batch_size: int, device: torch.device) -> torch.Tensor:
        b0 = F.softmax(self.planner_init_logits, dim=-1)
        return b0.unsqueeze(0).expand(batch_size, -1).to(device)

    def _belief_log(self, belief: torch.Tensor) -> torch.Tensor:
        return (belief + self.eps).log()

    def _apply_posterior_update(self, belief: torch.Tensor, token_logp_by_plan: torch.Tensor) -> torch.Tensor:
        # token_logp_by_plan: [B, K]
        return F.softmax(self._belief_log(belief) + token_logp_by_plan, dim=-1)

    def _apply_chunk_boundary_update(
        self,
        belief: torch.Tensor,
        transition: torch.Tensor,
        context_logit: torch.Tensor | None,
    ) -> torch.Tensor:
        prior = belief @ transition
        if context_logit is None:
            return prior
        return F.softmax(self._belief_log(prior) + context_logit, dim=-1)

    def _project_plan_states(self) -> torch.Tensor:
        return self.plan_proj(self.plan_emb.weight)  # [K, D]

    def _logp_per_plan(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns:
        # - log p(y_t | y_<t, z) for all z with shape [B, L, K]
        # - planner context logits from token history states with shape [B, L, K]
        x = self.token_emb(input_ids)
        h, _ = self.backbone(x)
        ctx_logits = self.cfg.planner_context_scale * self.planner_ctx_proj(h)  # [B, L, K]
        s = self.state_proj(h)  # [B, L, D]
        plan_repr = self._project_plan_states()  # [K, D]
        joint = torch.tanh(s.unsqueeze(2) + plan_repr.unsqueeze(0).unsqueeze(0))  # [B, L, K, D]
        logits = self.out_proj(self.dropout(joint)) + self.plan_vocab_bias.unsqueeze(0).unsqueeze(0)  # [B, L, K, V]
        log_probs = F.log_softmax(logits, dim=-1)
        idx = target_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_plan_states, 1)
        token_logp = torch.gather(log_probs, dim=-1, index=idx).squeeze(-1)
        return token_logp, ctx_logits

    def filtered_nll(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Exact filtered mixture over discrete planner states.
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        logp_per_plan, ctx_logits = self._logp_per_plan(input_ids, target_ids)  # [B, L, K], [B, L, K]

        P = self.transition_matrix()  # [K, K]
        belief = self.initial_belief(batch_size, device)  # [B, K]

        total_nll = torch.zeros((), device=device)
        belief_entropy_sum = torch.zeros((), device=device)
        belief_mass_sum = torch.zeros(self.num_plan_states, device=device)

        for t in range(seq_len):
            if t > 0 and t % self.chunk_size == 0:
                belief = self._apply_chunk_boundary_update(belief, P, ctx_logits[:, t, :])

            logp_z = logp_per_plan[:, t, :]  # [B, K]
            log_b = self._belief_log(belief)

            log_mix = torch.logsumexp(log_b + logp_z, dim=-1)  # [B]
            total_nll = total_nll - log_mix.sum()

            belief = self._apply_posterior_update(belief, logp_z)

            entropy = -(belief * (belief + self.eps).log()).sum(dim=-1)  # [B]
            belief_entropy_sum = belief_entropy_sum + entropy.mean()
            belief_mass_sum = belief_mass_sum + belief.sum(dim=0)

        nll = total_nll / (batch_size * seq_len)
        mean_entropy = belief_entropy_sum / seq_len
        mean_usage = belief_mass_sum / (batch_size * seq_len)
        uniform = torch.full_like(mean_usage, fill_value=1.0 / self.num_plan_states)
        usage_kl = torch.sum(mean_usage * ((mean_usage + self.eps).log() - uniform.log()))

        aux = {
            "belief_entropy": mean_entropy,
            "usage_kl_to_uniform": usage_kl,
        }
        return nll, aux

    @torch.inference_mode()
    def planner_lookahead(
        self,
        belief: torch.Tensor,
        steps: int,
        transition: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        P = self.transition_matrix() if transition is None else transition
        cur = belief
        outs: list[torch.Tensor] = []
        for _ in range(steps):
            cur = cur @ P
            outs.append(cur)
        return outs

    @torch.inference_mode()
    def generate_filtered(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
        lookahead_steps: int = 0,
    ) -> tuple[list[int], list[list[float]]]:
        self.eval()
        device = next(self.parameters()).device
        if len(prompt_ids) == 0:
            raise ValueError("prompt_ids cannot be empty")

        seq = list(prompt_ids)
        hidden: torch.Tensor | None = None

        belief = self.initial_belief(batch_size=1, device=device)
        P = self.transition_matrix()
        plan_repr = self._project_plan_states()
        inv_temp = 1.0 / max(temperature, 1e-6)

        # Update belief with observed prompt transitions (except first token).
        for idx in range(1, len(prompt_ids)):
            t = idx - 1
            prev_tok = torch.tensor([[prompt_ids[idx - 1]]], device=device, dtype=torch.long)
            emb = self.token_emb(prev_tok)
            out, hidden = self.backbone(emb, hidden)
            state = out[:, -1, :]  # [1, H]
            if t > 0 and t % self.chunk_size == 0:
                ctx_bias = self.cfg.planner_context_scale * self.planner_ctx_proj(state)
                belief = self._apply_chunk_boundary_update(belief, P, ctx_bias)
            logits_k = self._state_to_plan_logits(state, plan_repr=plan_repr)  # [1, K, V]
            logp_k = F.log_softmax(logits_k, dim=-1)[:, :, prompt_ids[idx]]  # [1, K]
            belief = self._apply_posterior_update(belief, logp_k)

        lookahead_trace: list[list[float]] = []
        token_pos = len(prompt_ids) - 1

        # Build state for next-token prediction from the last observed token.
        last_tok = torch.tensor([[prompt_ids[-1]]], device=device, dtype=torch.long)
        out, hidden = self.backbone(self.token_emb(last_tok), hidden)
        state_next = out[:, -1, :]

        for _ in range(max_new_tokens):
            if token_pos > 0 and token_pos % self.chunk_size == 0:
                ctx_bias = self.cfg.planner_context_scale * self.planner_ctx_proj(state_next)
                belief = self._apply_chunk_boundary_update(belief, P, ctx_bias)

            logits_k = self._state_to_plan_logits(state_next, plan_repr=plan_repr)  # [1, K, V]
            base_log_probs = F.log_softmax(logits_k, dim=-1).squeeze(0)  # [K, V]
            sample_log_probs = (
                F.log_softmax(logits_k * inv_temp, dim=-1).squeeze(0)
                if temperature != 1.0
                else base_log_probs
            )
            log_b = self._belief_log(belief.squeeze(0)).unsqueeze(-1)  # [K, 1]
            mix_log_probs = torch.logsumexp(log_b + sample_log_probs, dim=0)  # [V]

            if top_k is not None and top_k > 0 and top_k < mix_log_probs.numel():
                top_vals, top_idx = torch.topk(mix_log_probs, k=top_k)
                probs = F.softmax(top_vals, dim=-1)
                sample_local = torch.multinomial(probs, num_samples=1).item()
                next_id = int(top_idx[sample_local].item())
            else:
                probs = F.softmax(mix_log_probs, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            seq.append(next_id)

            logp_obs_k = base_log_probs[:, next_id].unsqueeze(0)
            belief = self._apply_posterior_update(belief, logp_obs_k)

            if lookahead_steps > 0:
                future = self.planner_lookahead(belief, lookahead_steps, transition=P)
                # Record first lookahead state for debugging.
                lookahead_trace.append(future[0].squeeze(0).detach().cpu().tolist())

            step_tok = torch.tensor([[next_id]], device=device, dtype=torch.long)
            out, hidden = self.backbone(self.token_emb(step_tok), hidden)
            state_next = out[:, -1, :]
            token_pos += 1
            if eos_id is not None and next_id == eos_id:
                break

        return seq, lookahead_trace

    def _state_to_plan_logits(
        self,
        state: torch.Tensor,
        *,
        plan_repr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # state: [B, H] -> logits [B, K, V]
        s = self.state_proj(state)  # [B, D]
        p = self._project_plan_states() if plan_repr is None else plan_repr
        joint = torch.tanh(s.unsqueeze(1) + p.unsqueeze(0))  # [B, K, D]
        return self.out_proj(self.dropout(joint)) + self.plan_vocab_bias.unsqueeze(0)
