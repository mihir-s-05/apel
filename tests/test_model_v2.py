import math
import pathlib
import sys
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))


class TestAPELRV2(unittest.TestCase):
    def test_sample_gumbel_state_bf16_is_finite_and_normalized(self) -> None:
        import torch

        from apelr.model_v2 import APELRV2Model, APELRV2ModelConfig

        torch.manual_seed(0)
        cfg = APELRV2ModelConfig(
            vocab_size=64,
            num_plan_states=4,
            num_experts=4,
            chunk_size=4,
            token_dim=16,
            hidden_dim=32,
            num_layers=1,
            dropout=0.0,
            async_planner=False,
        )
        model = APELRV2Model(cfg)
        logits = torch.randn(128, cfg.num_plan_states, dtype=torch.bfloat16)
        for hard_fraction in (0.0, 0.5, 1.0):
            for _ in range(8):
                state = model._sample_gumbel_state(logits, tau=1.0, hard_fraction=hard_fraction)
                self.assertEqual(state.dtype, logits.dtype)
                self.assertTrue(torch.isfinite(state).all().item())
                row_sums = state.float().sum(dim=-1)
                self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=1e-3))

    def test_compute_losses_runs_and_backprop(self) -> None:
        import torch

        from apelr.model_v2 import APELRV2Model, APELRV2ModelConfig

        torch.manual_seed(0)
        cfg = APELRV2ModelConfig(
            vocab_size=101,
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
        model.train()
        x = torch.randint(0, cfg.vocab_size, (2, 32), dtype=torch.long)
        y = torch.randint(0, cfg.vocab_size, (2, 32), dtype=torch.long)
        nll, aux = model.compute_losses(x, y, planner_temperature=1.0)
        loss = (
            nll
            + 0.1 * aux["usage_kl_to_uniform"]
            + 0.1 * aux["boundary_entropy"]
            + 0.1 * aux["future_contrastive_loss"]
            - 0.1 * aux["plan_js_div_loss"]
        )
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            self.assertTrue(torch.isfinite(param.grad).all().item(), msg=f"Non-finite grad in {name}")
            break

    def test_compute_losses_validations(self) -> None:
        import torch

        from apelr.model_v2 import APELRV2Model, APELRV2ModelConfig

        cfg = APELRV2ModelConfig(vocab_size=32, num_plan_states=2, num_experts=2, chunk_size=4, token_dim=8, hidden_dim=16, num_layers=1, dropout=0.0)
        model = APELRV2Model(cfg)
        x = torch.zeros((1, 8), dtype=torch.long)
        y = torch.zeros((1, 7), dtype=torch.long)
        with self.assertRaises(ValueError):
            model.compute_losses(x, y)
        with self.assertRaises(ValueError):
            model.compute_losses(torch.zeros((1, 8), dtype=torch.long), torch.zeros((1, 8), dtype=torch.long), planner_mode="nope")

    def test_generate_planned_validations(self) -> None:
        import torch

        from apelr.model_v2 import APELRV2Model, APELRV2ModelConfig

        torch.manual_seed(0)
        cfg = APELRV2ModelConfig(vocab_size=32, num_plan_states=2, num_experts=2, chunk_size=4, token_dim=8, hidden_dim=16, num_layers=1, dropout=0.0)
        model = APELRV2Model(cfg)
        with self.assertRaises(ValueError):
            model.generate_planned(prompt_ids=[1], max_new_tokens=1, top_p=0.0)
        with self.assertRaises(ValueError):
            model.generate_planned(prompt_ids=[1], max_new_tokens=1, force_state=5)
        with self.assertRaises(ValueError):
            model.generate_planned(prompt_ids=[1], max_new_tokens=-1)

    def test_generate_planned_runs(self) -> None:
        import torch

        from apelr.model_v2 import APELRV2Model, APELRV2ModelConfig

        torch.manual_seed(0)
        cfg = APELRV2ModelConfig(
            vocab_size=64,
            num_plan_states=4,
            num_experts=4,
            chunk_size=4,
            token_dim=16,
            hidden_dim=32,
            num_layers=1,
            dropout=0.0,
            token_filtering=True,
            async_planner=False,
        )
        model = APELRV2Model(cfg)
        out, trace = model.generate_planned(
            prompt_ids=[1, 2, 3, 4],
            max_new_tokens=5,
            temperature=1.0,
            top_k=None,
            top_p=1.0,
            lookahead_steps=2,
            async_planner=False,
        )
        self.assertEqual(len(out), 4 + 5)
        self.assertEqual(len(trace), 5)
        self.assertTrue(all(isinstance(v, int) for v in out))
        self.assertTrue(all(isinstance(row, list) for row in trace))
        self.assertTrue(all(math.isfinite(float(x)) for row in trace for x in row))


if __name__ == "__main__":
    unittest.main()

