from methods.success_replay import make_success_replay_method


METHOD = make_success_replay_method(
    "bc_priority",
    teacher="best",
    gradient_strategy="pcgrad_bc_priority",
)
