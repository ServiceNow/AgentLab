import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

CheatCustomFn = Callable[..., list[str]]

CHEAT_CUSTOM_REGISTRY: dict[type, CheatCustomFn] = {}
_REGISTRY_READY = False


def register_cheat_custom(task_cls: type, fn: CheatCustomFn) -> None:
    """Register a cheat_custom implementation for a specific task class."""
    CHEAT_CUSTOM_REGISTRY[task_cls] = fn


def _task_id(task_cls: type) -> str | None:
    getter = getattr(task_cls, "get_task_id", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception:
            return None
    return None


def _make_stub(task_cls: type) -> CheatCustomFn:
    task_id = _task_id(task_cls)
    cls_name = task_cls.__name__

    def cheat_custom(self, page=None, chat_messages=None, subtask_idx=None):
        msg = f"cheat_custom not implemented for {cls_name}"
        if task_id:
            msg += f" (task_id={task_id})"
        raise NotImplementedError(msg)

    return cheat_custom


def register_workarena_stubs() -> bool:
    """Populate stubs for all known WorkArena tasks (best-effort)."""
    try:
        import browsergym.workarena as wa

        tasks = getattr(wa, "ALL_WORKARENA_TASKS", [])
    except Exception as exc:
        logger.warning("Could not import WorkArena tasks for stubs: %s", exc)
        return False

    for task_cls in tasks:
        if task_cls in CHEAT_CUSTOM_REGISTRY:
            continue
        if hasattr(task_cls, "cheat_custom"):
            # Respect any existing implementation.
            continue
        CHEAT_CUSTOM_REGISTRY[task_cls] = _make_stub(task_cls)

    return True


def _ensure_registry_ready() -> None:
    global _REGISTRY_READY
    if _REGISTRY_READY:
        return
    _REGISTRY_READY = True
    register_workarena_stubs()
    try:
        from agentlab.cheat_custom.workarena_adapters import (
            register_workarena_cheat_customs,
        )

        register_workarena_cheat_customs()
    except ImportError:
        logger.debug("No WorkArena cheat_custom adapters found.")
    except Exception as exc:
        logger.warning("Failed to register WorkArena cheat_custom adapters: %s", exc)


def ensure_cheat_custom(task: Any) -> CheatCustomFn | None:
    """Attach a cheat_custom method to the task (stub or registered impl).

    Returns the bound cheat_custom if available.
    """
    if task is None:
        return None

    _ensure_registry_ready()

    if hasattr(task, "cheat_custom"):
        return getattr(task, "cheat_custom")

    task_cls = type(task)
    fn = CHEAT_CUSTOM_REGISTRY.get(task_cls)
    if fn is None:
        fn = _make_stub(task_cls)
        CHEAT_CUSTOM_REGISTRY[task_cls] = fn

    try:
        setattr(task_cls, "cheat_custom", fn)
        return getattr(task, "cheat_custom")
    except Exception:
        # Fallback: attach to instance only.
        try:
            import types

            bound = types.MethodType(fn, task)
            setattr(task, "cheat_custom", bound)
            return bound
        except Exception as exc:
            logger.warning("Failed to attach cheat_custom to %s: %s", task_cls, exc)
            return None
