from methods.success_replay import make_success_replay_method


METHOD = make_success_replay_method(
    "no_projection",
    teacher="best",
)
