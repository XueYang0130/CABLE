from typing import Any

from agents import PackNetSACAgent, SACAgent
from methods.base import MethodSpec


def build_agent(
    agent_kwargs: dict[str, Any],
    args: Any,
    total_tasks: int,
) -> SACAgent:
    return PackNetSACAgent(
        **agent_kwargs,
        total_tasks=total_tasks,
        retrain_steps=args.packnet_retrain_steps,
    )


METHOD = MethodSpec(
    method_id="packnet",
    modes=frozenset({"continual"}),
    append_task_id=True,
    multi_head=True,
    hide_task_id=True,
    reference_exploration=True,
    defaults={
        "packnet_retrain_steps": 0,
        "gradient_clip_norm": 0.1,
    },
    agent_factory=build_agent,
)
