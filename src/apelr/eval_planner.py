from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model_v2 import APELRV2Model


@torch.inference_mode()
def evaluate_planner_usage_v2(
    model: APELRV2Model,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
    *,
    commitment: str = "soft",
    planner_temperature: float = 1.0,
) -> dict[str, float]:
    model.eval()
    deltas: list[float] = []
    divergences: list[float] = []
    persistence: list[float] = []
    utilization: list[float] = []
    nlls: list[float] = []
    future_losses: list[float] = []
    js_losses: list[float] = []
    feedback_deltas: list[float] = []

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        diag = model.planner_usage_metrics(
            x,
            y,
            commitment=commitment,
            planner_temperature=planner_temperature,
        )
        nll_with_feedback, aux = model.compute_losses(
            x,
            y,
            planner_mode="normal",
            commitment=commitment,
            planner_temperature=planner_temperature,
            lookahead_steps=int(model.cfg.lookahead_horizon),
            lookahead_feedback_scale=float(model.cfg.lookahead_feedback_scale),
        )
        nll_no_feedback, _ = model.compute_losses(
            x,
            y,
            planner_mode="normal",
            commitment=commitment,
            planner_temperature=planner_temperature,
            lookahead_steps=int(model.cfg.lookahead_horizon),
            lookahead_feedback_scale=0.0,
        )
        deltas.append(float(diag["planner_mask_delta_loss"].item()))
        divergences.append(float(diag["forced_state_divergence"].item()))
        persistence.append(float(diag["state_persistence"].item()))
        utilization.append(float(diag["expert_utilization"].item()))
        nlls.append(float(nll_with_feedback.item()))
        future_losses.append(float(aux["future_contrastive_loss"].item()))
        js_losses.append(float(aux["plan_js_div_loss"].item()))
        feedback_deltas.append(float((nll_no_feedback - nll_with_feedback).item()))

    if not nlls:
        return {
            "loss": float("nan"),
            "planner_mask_delta_loss": float("nan"),
            "forced_state_divergence": float("nan"),
            "state_persistence": float("nan"),
            "expert_utilization": float("nan"),
            "future_contrastive_loss": float("nan"),
            "plan_js_div_loss": float("nan"),
            "feedback_delta_loss": float("nan"),
        }

    return {
        "loss": float(np.mean(nlls)),
        "planner_mask_delta_loss": float(np.mean(deltas)),
        "forced_state_divergence": float(np.mean(divergences)),
        "state_persistence": float(np.mean(persistence)),
        "expert_utilization": float(np.mean(utilization)),
        "future_contrastive_loss": float(np.mean(future_losses)),
        "plan_js_div_loss": float(np.mean(js_losses)),
        "feedback_delta_loss": float(np.mean(feedback_deltas)),
    }


def planner_usage_summary(metrics: dict[str, Any]) -> str:
    return (
        f"loss={metrics.get('loss', float('nan')):.4f} "
        f"mask_delta={metrics.get('planner_mask_delta_loss', float('nan')):.4f} "
        f"force_js={metrics.get('forced_state_divergence', float('nan')):.4f} "
        f"state_persist={metrics.get('state_persistence', float('nan')):.4f} "
        f"expert_util={metrics.get('expert_utilization', float('nan')):.4f} "
        f"fcl={metrics.get('future_contrastive_loss', float('nan')):.4f} "
        f"js={metrics.get('plan_js_div_loss', float('nan')):.4f} "
        f"fb_d={metrics.get('feedback_delta_loss', float('nan')):.4f}"
    )
