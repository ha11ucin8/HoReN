_GENERATION_CALL_COUNT = 0


def reset_generation_call_count():
    global _GENERATION_CALL_COUNT
    _GENERATION_CALL_COUNT = 0


def increment_generation_call_count(n: int = 1):
    global _GENERATION_CALL_COUNT
    _GENERATION_CALL_COUNT += int(n)


def get_generation_call_count() -> int:
    return int(_GENERATION_CALL_COUNT)
