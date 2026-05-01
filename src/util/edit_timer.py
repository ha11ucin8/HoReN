"""
Global accumulator for pure model-editing time.

Each editor's core computation calls add_edit_time() with the elapsed
seconds for its training/update loop, excluding initialization overhead
(parameter freezing, layer-swapping, tokenization, weight backup, etc.).

Usage in editor.py:
    reset_edit_timer()
    edited_model, weights_copy, icl_examples = edit_func(request)
    pure_time = get_edit_time()   # 0.0 if editor is not instrumented
"""

_PURE_EDIT_TIME: float = 0.0


def reset_edit_timer() -> None:
    global _PURE_EDIT_TIME
    _PURE_EDIT_TIME = 0.0


def add_edit_time(elapsed: float) -> None:
    global _PURE_EDIT_TIME
    _PURE_EDIT_TIME += elapsed


def get_edit_time() -> float:
    return _PURE_EDIT_TIME
