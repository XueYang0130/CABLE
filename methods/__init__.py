from methods.base import MethodSpec
from methods.fine_tuning import METHOD as FINE_TUNING
from methods.clonex_sac import METHOD as CLONEX_SAC
from methods.packnet import METHOD as PACKNET
from methods.recall import METHOD as RECALL
from methods.success_replay_best import METHOD as SUCCESS_REPLAY_BEST
from methods.success_replay_best_pcgrad import METHOD as SUCCESS_REPLAY_BEST_PCGRAD
from methods.success_replay_final_pcgrad import METHOD as SUCCESS_REPLAY_FINAL_PCGRAD
from methods.success_replay_best_bc_priority_pcgrad import METHOD as SUCCESS_REPLAY_BEST_BC_PRIORITY_PCGRAD
from methods.all_replay_best_pcgrad import METHOD as ALL_REPLAY_BEST_PCGRAD

SINGLE_TASK_BASELINE = MethodSpec(method_id="single_task_baseline", modes=frozenset({"single", "single-batch"}))
METHOD_REGISTRY = {m.method_id: m for m in [SINGLE_TASK_BASELINE, FINE_TUNING, CLONEX_SAC, PACKNET, RECALL, SUCCESS_REPLAY_BEST, SUCCESS_REPLAY_BEST_PCGRAD, SUCCESS_REPLAY_FINAL_PCGRAD, SUCCESS_REPLAY_BEST_BC_PRIORITY_PCGRAD, ALL_REPLAY_BEST_PCGRAD]}
BC_GRADIENT_STRATEGY_BY_METHOD = {
    "success_replay_best_pcgrad": "pcgrad_sac_priority",
    "success_replay_final_pcgrad": "pcgrad_sac_priority",
    "all_replay_best_pcgrad": "pcgrad_sac_priority",
    "success_replay_best_bc_priority_pcgrad": "pcgrad_bc_priority",
}
BC_COMBINATION_STRATEGY_BY_METHOD = {}

def get_method(method_id):
    try:
        return METHOD_REGISTRY[method_id]
    except KeyError as error:
        raise ValueError(f"Unsupported method: {method_id}") from error

def available_method_ids():
    return list(METHOD_REGISTRY)

def default_method_for_mode(mode):
    if mode in {"single", "single-batch"}:
        return "single_task_baseline"
    if mode == "continual":
        return "fine_tuning"
    raise ValueError(f"Unsupported mode: {mode}")

def method_defaults(method_id):
    return dict(get_method(method_id).defaults)

def bc_gradient_strategy_for_method(method_id):
    get_method(method_id)
    return BC_GRADIENT_STRATEGY_BY_METHOD.get(method_id, "standard")

def required_bc_combination_strategy(method_id):
    get_method(method_id)
    return None

def is_method_compatible(mode, method_id):
    return mode in get_method(method_id).modes
