import math


def forensic_response():
    summary = {
        "output_token_count": 3,
        "grammar_active_token_count": 3,
        "finite_active_token_count": 3,
        "cumulative_constraint_pressure_nats": 1.6,
        "mean_constraint_pressure_nats": 1.6 / 3,
        "max_constraint_pressure_nats": 1.5,
        "max_pressure_output_index": 1,
        "argmax_masked_count": 1,
        "mean_top1_logit_gap_to_top2_nats": 0.5,
        "max_top1_logit_gap_to_top2_nats": 0.8,
        "high_pressure_low_entropy_count": 1,
        "longest_high_pressure_run": 1,
        "empty_support_count": 0,
        "eos_only_support_count": 0,
        "has_non_finite_count": 0,
        "effective_legal_token_count_histogram": [0, 1, 1],
        "effective_legal_token_count_histogram_labels": ["0", "1", "2-3"],
        "top_pressure_events": [
            {
                "output_index": 1,
                "constraint_pressure_nats": 1.5,
                "effective_legal_token_count": 2,
                "emitted_token_id": 11,
                "unconstrained_top1_id": 21,
                "top1_logit_gap_to_top2_nats": 0.8,
                "argmax_masked": True,
                "flags": [],
                "spec_source": "accepted_draft",
            }
        ],
    }
    tokens = {
        "count": 3,
        "output_index": [0, 1, 2],
        "token_id": [10, 11, 12],
        "grammar_active": [True, True, True],
        "effective_legal_token_count": [3, 2, 1],
        "constraint_pressure_nats": [0.1, 1.5, 0.0],
        "argmax_masked": [False, True, False],
        "unconstrained_top1": {
            "token_id": [10, 21, 12],
            "prob": [0.6, 0.7, 0.8],
            "logit_gap_to_top2_nats": [0.2, 0.8, 0.5],
        },
        "entropy_nats": [0.9, 0.2, 0.3],
        "emitted": {
            "allowed": [True, True, True],
            "model_log_prob": [math.log(0.5), math.log(0.1), math.log(0.8)],
            "rank": [1, 4, 1],
        },
        "flags": {
            "grammar_inactive": [False, False, False],
            "empty_support": [False, False, False],
            "all_allow": [False, False, False],
            "eos_only_support": [False, False, False],
            "has_non_finite_logits": [False, False, False],
        },
        "spec_source": [None, "accepted_draft", None],
        "top_masked": {
            "width": 2,
            "count": [1, 2, 0],
            "token_id": [30, -1, 31, 32, -1, -1],
            "prob": [0.3, 0.0, 0.4, 0.2, 0.0, 0.0],
        },
        "top_legal_after_conditioning": {
            "width": 2,
            "count": [2, 1, 1],
            "token_id": [10, 33, 11, -1, 12, -1],
            "constrained_prob": [0.55, 0.25, 0.45, 0.0, 0.8, 0.0],
        },
    }
    return {
        "text": "alpha beta gamma",
        "meta_info": {
            "constraint_metrics": {
                "status": "complete",
                "mode": "forensic",
                "warnings": [],
                "summary": summary,
                "input_token_count": 5,
                "output_token_count": 3,
                "tokens": tokens,
            }
        },
    }
