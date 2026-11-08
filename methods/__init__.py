from methods.base import MethodSpec
from methods.fine_tuning import METHOD as FINE_TUNING
from methods.clonex_sac import METHOD as CLONEX_SAC
from methods.packnet import METHOD as PACKNET
from methods.recall import METHOD as RECALL
from methods.no_projection import METHOD as NO_PROJECTION
from methods.cable import METHOD as CABLE
from methods.final_teacher import METHOD as FINAL_TEACHER
from methods.bc_priority import METHOD as BC_PRIORITY
from methods.all_states import METHOD as ALL_STATES

SINGLE_TASK_BASELINE = MethodSpec(method_id="single_task_baseline", modes=frozenset({"single", "single-batch"}))
METHOD_REGISTRY = {m.method_id: m for m in [SINGLE_TASK_BASELINE, FINE_TUNING, CLONEX_SAC, PACKNET, RECALL, NO_PROJECTION, CABLE, FINAL_TEACHER, BC_PRIORITY, ALL_STATES]}
BC_GRADIENT_STRATEGY_BY_METHOD = {
    "cable": "pcgrad_sac_priority",
    "final_teacher": "pcgrad_sac_priority",
    "all_states": "pcgrad_sac_priority",
    "bc_priority": "pcgrad_bc_priority",
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
