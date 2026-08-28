"""Packed constraint-metric row layout shared across runtime stages."""

FLOAT_FIELDS = (
    "constraint_pressure_nats",
    "entropy_nats",
    "unconstrained_top1_prob",
    "unconstrained_top1_logit_gap_to_top2_nats",
    "emitted_model_log_prob",
)
FLOAT_PRESSURE = FLOAT_FIELDS.index("constraint_pressure_nats")
FLOAT_ENTROPY = FLOAT_FIELDS.index("entropy_nats")
FLOAT_UNCONSTRAINED_TOP1_PROB = FLOAT_FIELDS.index("unconstrained_top1_prob")
FLOAT_TOP1_LOGIT_GAP = FLOAT_FIELDS.index("unconstrained_top1_logit_gap_to_top2_nats")
FLOAT_EMITTED_MODEL_LOG_PROB = FLOAT_FIELDS.index("emitted_model_log_prob")

BOOL_FIELDS = (
    "grammar_active",
    "argmax_masked",
    "empty_support",
    "eos_only_support",
    "has_non_finite_logits",
    "emitted_allowed",
)
BOOL_ARGMAX_MASKED = BOOL_FIELDS.index("argmax_masked")
BOOL_EMPTY_SUPPORT = BOOL_FIELDS.index("empty_support")
BOOL_EOS_ONLY_SUPPORT = BOOL_FIELDS.index("eos_only_support")
BOOL_HAS_NON_FINITE_LOGITS = BOOL_FIELDS.index("has_non_finite_logits")
BOOL_EMITTED_ALLOWED = BOOL_FIELDS.index("emitted_allowed")

SPEC_SOURCE_NORMAL = 0
SPEC_SOURCE_ACCEPTED_DRAFT = 1
SPEC_SOURCE_BONUS = 2
SPEC_SOURCE_NAMES = {
    SPEC_SOURCE_ACCEPTED_DRAFT: "accepted_draft",
    SPEC_SOURCE_BONUS: "bonus",
}

# The fused kernel writes the six scalar integer fields first, then every bool
# except grammar_active (which is already request metadata), followed by two K-wide
# token-id tables. Floats contain FLOAT_FIELDS followed by two K-wide probability
# tables. These counts are also the candidate-table offsets.
KERNEL_INT_FIELDS = (
    "unconstrained_top1_token_id",
    "emitted_rank",
    "effective_legal_token_count",
    "blocked_token_count",
    "token_id",
    "spec_source",
)
KERNEL_INT_UNCONSTRAINED_TOP1_TOKEN_ID = KERNEL_INT_FIELDS.index(
    "unconstrained_top1_token_id"
)
KERNEL_INT_EMITTED_RANK = KERNEL_INT_FIELDS.index("emitted_rank")
KERNEL_INT_EFFECTIVE_LEGAL_TOKEN_COUNT = KERNEL_INT_FIELDS.index(
    "effective_legal_token_count"
)
KERNEL_INT_BLOCKED_TOKEN_COUNT = KERNEL_INT_FIELDS.index("blocked_token_count")
KERNEL_INT_TOKEN_ID = KERNEL_INT_FIELDS.index("token_id")
KERNEL_INT_SPEC_SOURCE = KERNEL_INT_FIELDS.index("spec_source")
KERNEL_VALUE_INT_COUNT = len(KERNEL_INT_FIELDS)
KERNEL_BOOL_COUNT = len(BOOL_FIELDS) - 1
KERNEL_ROW_INT_COUNT = KERNEL_VALUE_INT_COUNT + KERNEL_BOOL_COUNT
KERNEL_INT_ARGMAX_MASKED = KERNEL_VALUE_INT_COUNT + BOOL_ARGMAX_MASKED - 1
KERNEL_INT_EMPTY_SUPPORT = KERNEL_VALUE_INT_COUNT + BOOL_EMPTY_SUPPORT - 1
KERNEL_INT_EOS_ONLY_SUPPORT = KERNEL_VALUE_INT_COUNT + BOOL_EOS_ONLY_SUPPORT - 1
KERNEL_INT_HAS_NON_FINITE_LOGITS = (
    KERNEL_VALUE_INT_COUNT + BOOL_HAS_NON_FINITE_LOGITS - 1
)
KERNEL_INT_EMITTED_ALLOWED = KERNEL_VALUE_INT_COUNT + BOOL_EMITTED_ALLOWED - 1
KERNEL_ROW_FLOAT_COUNT = len(FLOAT_FIELDS)
TOP_K_INT_GROUP_COUNT = 2
TOP_K_FLOAT_GROUP_COUNT = 2
