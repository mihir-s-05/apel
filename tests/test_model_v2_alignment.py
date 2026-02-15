import pathlib
import sys
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))


class TestAPELRV2Alignment(unittest.TestCase):
    def _teacher_forced_nll(
        self,
        *,
        model,
        x,
        y,
        planner_temperature: float,
        lookahead_steps: int,
        lookahead_feedback_scale: float,
        token_filtering: bool,
    ):
        import torch
        import torch.nn.functional as F

        model.eval()
        device = next(model.parameters()).device
        x = x.to(device)
        y = y.to(device)

        trans = model.transition_matrix()
        belief = model._initial_belief(1, device)
        hidden = None
        chunk_hidden_buffer = []
        token_pos = 0

        def maybe_boundary_update():
            nonlocal belief, chunk_hidden_buffer
            if len(chunk_hidden_buffer) == 0:
                return
            if token_pos % model.chunk_size != 0:
                return
            chunk_mean = torch.stack(chunk_hidden_buffer, dim=1).mean(dim=1)
            belief = model._apply_boundary_update(belief, trans, chunk_mean)
            chunk_hidden_buffer = []

        tok0 = int(x[0].item())
        tok_t = torch.tensor([[tok0]], device=device, dtype=torch.long)
        out, hidden = model.backbone(model.token_emb(tok_t), hidden)
        chunk_hidden_buffer.append(out[:, -1, :])
        token_pos += 1
        maybe_boundary_update()

        nll_steps = []
        for t in range(int(y.numel())):
            state = (
                hidden[-1, :, :]
                if hidden is not None
                else torch.zeros((1, model.cfg.hidden_dim), device=device)
            )
            expert_logits = torch.stack([head(state) for head in model.expert_heads], dim=1)
            expert_log_probs = F.log_softmax(expert_logits, dim=-1).squeeze(0)

            gate, _ = model._planner_gate_with_lookahead(
                belief,
                trans,
                planner_temperature=float(planner_temperature),
                lookahead_steps=int(lookahead_steps),
                feedback_scale=float(lookahead_feedback_scale),
            )
            mix_log = torch.logsumexp((gate + model.eps).log().squeeze(0).unsqueeze(-1) + expert_log_probs, dim=0)

            tgt = int(y[t].item())
            nll_steps.append(-mix_log[tgt])

            if token_filtering:
                obs_logp = expert_log_probs[:, tgt].unsqueeze(0)
                belief = F.softmax(model._belief_log(belief) + obs_logp, dim=-1)

            tok_t = torch.tensor([[tgt]], device=device, dtype=torch.long)
            out, hidden = model.backbone(model.token_emb(tok_t), hidden)
            chunk_hidden_buffer.append(out[:, -1, :])
            token_pos += 1
            maybe_boundary_update()

        return torch.stack(nll_steps).mean()

    def test_compute_losses_matches_stepwise_teacher_forcing(self) -> None:
        import torch

        from apelr.model_v2 import APELRV2Model, APELRV2ModelConfig

        torch.manual_seed(0)
        cfg = APELRV2ModelConfig(
            vocab_size=97,
            num_plan_states=4,
            num_experts=4,
            chunk_size=8,
            token_dim=32,
            hidden_dim=64,
            num_layers=1,
            dropout=0.0,
            lookahead_horizon=2,
            lookahead_feedback_scale=0.25,
            token_filtering=True,
            async_planner=False,
        )
        model = APELRV2Model(cfg)
        model.eval()

        seq_len = 32
        stream = torch.randint(0, cfg.vocab_size, (seq_len + 1,), dtype=torch.long)
        x = stream[:-1]
        y = stream[1:]

        planner_temperature = 1.1
        lookahead_steps = 2
        feedback_scale = 0.25

        nll_batch, _aux = model.compute_losses(
            x.unsqueeze(0),
            y.unsqueeze(0),
            planner_mode="normal",
            commitment="soft",
            planner_temperature=planner_temperature,
            lookahead_steps=lookahead_steps,
            lookahead_feedback_scale=feedback_scale,
            token_filtering=True,
        )
        nll_step = self._teacher_forced_nll(
            model=model,
            x=x,
            y=y,
            planner_temperature=planner_temperature,
            lookahead_steps=lookahead_steps,
            lookahead_feedback_scale=feedback_scale,
            token_filtering=True,
        )
        self.assertLess(float((nll_batch - nll_step).abs()), 1e-5)


if __name__ == "__main__":
    unittest.main()

