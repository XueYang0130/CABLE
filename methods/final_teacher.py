from methods.success_replay import make_success_replay_method


METHOD = make_success_replay_method(
    "final_teacher",
    teacher="final",
    gradient_strategy="pcgrad_sac_priority",
)
