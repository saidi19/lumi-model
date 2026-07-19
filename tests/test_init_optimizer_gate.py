from __future__ import annotations

from copy import deepcopy

from lumi.init_optimizer_gate import evaluate_ab_decision


def _specification() -> dict:
    return {
        "training": {"seeds": [7, 17, 29]},
        "decision": {
            "max_regressing_seeds": 1,
            "max_seed_heldout_regression_fraction": 0.001,
            "max_throughput_regression_fraction": 0.1,
            "min_median_heldout_improvement_fraction": 0.002,
        },
        "variants": [
            {
                "initialization_strategy": "standard",
                "label": "standard_legacy",
                "optimizer_profile": "legacy",
            },
            {
                "initialization_strategy": "residual_scaled_v0.1",
                "label": "residual_legacy",
                "optimizer_profile": "legacy",
            },
            {
                "initialization_strategy": "standard",
                "label": "standard_grouped",
                "optimizer_profile": "grouped_adamw_beta2_0.95_v0.1",
            },
            {
                "initialization_strategy": "residual_scaled_v0.1",
                "label": "residual_grouped",
                "optimizer_profile": "grouped_adamw_beta2_0.95_v0.1",
            },
        ],
    }


def _results(losses: dict[str, list[float]], throughputs: dict[str, float]) -> list[dict]:
    results = []
    for variant, values in losses.items():
        for seed, loss in zip((7, 17, 29), values, strict=True):
            results.append(
                {
                    "heldout": {"final": {"loss": loss}},
                    "seed": seed,
                    "training_metrics": {
                        "final_loss": loss - 0.5,
                        "median_tokens_per_second": throughputs[variant],
                    },
                    "variant": variant,
                }
            )
    return results


def test_decision_promotes_clear_combined_winner() -> None:
    losses = {
        "standard_legacy": [5.0, 5.1, 4.9],
        "residual_legacy": [4.98, 5.08, 4.88],
        "standard_grouped": [4.97, 5.07, 4.87],
        "residual_grouped": [4.8, 4.9, 4.7],
    }
    throughputs = {label: 100.0 for label in losses}

    result = evaluate_ab_decision(_specification(), _results(losses, throughputs))

    assert result["decision"] == {
        "initialization_strategy": "residual_scaled_v0.1",
        "optimizer_profile": "grouped_adamw_beta2_0.95_v0.1",
        "reason": "clear_proxy_improvement",
        "winner": "residual_grouped",
    }


def test_decision_keeps_baseline_when_gain_is_too_small() -> None:
    losses = {
        "standard_legacy": [5.0, 5.1, 4.9],
        "residual_legacy": [4.999, 5.099, 4.899],
        "standard_grouped": [5.0, 5.1, 4.9],
        "residual_grouped": [4.999, 5.099, 4.899],
    }
    throughputs = {label: 100.0 for label in losses}

    result = evaluate_ab_decision(_specification(), _results(losses, throughputs))

    assert result["decision"]["winner"] == "standard_legacy"
    assert result["decision"]["reason"] == "no_candidate_cleared_conservative_thresholds"


def test_decision_rejects_fast_loss_winner_with_slow_throughput() -> None:
    specification = deepcopy(_specification())
    losses = {
        "standard_legacy": [5.0, 5.1, 4.9],
        "residual_legacy": [5.0, 5.1, 4.9],
        "standard_grouped": [5.0, 5.1, 4.9],
        "residual_grouped": [4.8, 4.9, 4.7],
    }
    throughputs = {label: 100.0 for label in losses}
    throughputs["residual_grouped"] = 80.0

    result = evaluate_ab_decision(specification, _results(losses, throughputs))

    assert result["comparisons_to_baseline"]["residual_grouped"]["eligible"] is False
    assert result["decision"]["winner"] == "standard_legacy"
