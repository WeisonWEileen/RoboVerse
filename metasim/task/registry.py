from __future__ import annotations

import pkgutil
from importlib import import_module

from loguru import logger as log

from metasim.task.base import BaseTaskEnv

# Global registry mapping lowercase names to task wrapper classes
TASK_REGISTRY = {}

# Global registry mapping lowercase names to task config classes
TASK_CFG_REGISTRY = {}


def register_task(*names):
    """Class decorator to register a task under one or more names.

    Usage:
        @register_task("humanoid.walk", "walk")
        class WalkTask(...):
            ...
    """
    if not names:
        raise ValueError("At least one name must be provided to register_task().")

    def _decorator(cls):
        # if not issubclass(cls, BaseTaskEnv):
        #     raise TypeError(f"Can only register subclasses of BaseTaskEnv, got: {cls!r}")
        for raw_name in names:
            key = raw_name.strip().lower()
            if not key:
                log.warning(f"Register class {cls!r} is not a subclass of BaseTaskEnv")
            existing = TASK_REGISTRY.get(key)
            if existing is not None and existing is not cls:
                raise ValueError(f"Task name '{key}' is already registered to {existing.__name__}.")
            TASK_REGISTRY[key] = cls
        return cls

    return _decorator


def _discover_task_modules() -> None:
    """Import modules from known task packages so @register_task runs.

    Scans these packages (if available):
      - metasim.example.example_pack.tasks
      - roboverse_pack.tasks


    Safe to call multiple times; import errors are ignored to avoid breaking
    discovery due to one bad module.
    """
    packages_to_scan = [
        "metasim.example.example_pack.tasks",
        "roboverse_pack.tasks",
        # project-local wrappers/tasks
        "humanoid_visualrl.wrapper",
    ]

    for pkg_name in packages_to_scan:
        try:
            # Import the root package
            pkg = import_module(pkg_name)
        except Exception as e:
            log.error(f"Task discovery: failed to import package '{pkg_name}': {e}")
            continue

        try:
            pkg_path = getattr(pkg, "__path__", None)
            if pkg_path is None:
                continue

            # Scan and import all submodules
            for _finder, module_name, _is_pkg in pkgutil.walk_packages(pkg_path, prefix=pkg.__name__ + "."):
                try:
                    import_module(module_name)
                except Exception as e:
                    log.error(f"Task discovery: failed to import module '{module_name}': {e}")
        except Exception as e:
            log.error(f"Task discovery: error scanning package '{pkg_name}': {e}")


def _discover_cfg_modules() -> None:
    """Import modules from known config packages so config registrations run.

    Scans these packages (if available):
      - humanoid_visualrl.cfg
      - roboverse_pack.tasks

    Safe to call multiple times; import errors are logged and ignored.
    """
    packages_to_scan = [
        "humanoid_visualrl.cfg",
        "roboverse_pack.tasks",
    ]

    for pkg_name in packages_to_scan:
        try:
            pkg = import_module(pkg_name)
        except Exception as e:
            log.error(f"Task cfg discovery: failed to import package '{pkg_name}': {e}")
            continue

        try:
            pkg_path = getattr(pkg, "__path__", None)
            if pkg_path is None:
                # Not a package, try importing as a single module
                continue

            for _finder, module_name, _is_pkg in pkgutil.walk_packages(pkg_path, prefix=pkg.__name__ + "."):
                try:
                    import_module(module_name)
                except Exception as e:
                    log.error(f"Task cfg discovery: failed to import module '{module_name}': {e}")
        except Exception as e:
            log.error(f"Task cfg discovery: error scanning package '{pkg_name}': {e}")


def get_task_class(name: str) -> type[BaseTaskEnv]:
    """Return the task wrapper class registered under the given name.

    Name lookup is case-insensitive.
    """
    # ensure modules are imported so registry is populated (idempotent)
    _discover_task_modules()

    key = name.strip().lower()
    try:
        return TASK_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_REGISTRY.keys())) or "<none>"
        raise KeyError(f"Unknown task '{name}'. Available tasks: {available}") from exc


def list_tasks():
    """List all registered task names (sorted)."""
    _discover_task_modules()
    return sorted(TASK_REGISTRY.keys())


def register_task_cfg(*names):
    """Class decorator to register a task config under one or more names.

    Usage:
        @register_task_cfg("humanoid.walk", "walk")
        @configclass  # optional; will be applied automatically if omitted
        class WalkCfg(...):
            ...
    """
    if not names:
        raise ValueError("At least one name must be provided to register_task_cfg().")

    def _decorator(cls):
        # If not already a dataclass/configclass, try to apply configclass to ensure consistency.
        if not hasattr(cls, "__dataclass_fields__"):
            try:
                # Local import to avoid circular import at module load time
                from metasim.utils.configclass import configclass as _configclass

                cls = _configclass(cls)
            except Exception:
                # If wrapping fails, proceed without altering the class
                pass
        for raw_name in names:
            key = raw_name.strip().lower()
            if not key:
                log.warning(f"Register config class {cls!r} has empty name entry; skipping")
                continue
            existing = TASK_CFG_REGISTRY.get(key)
            if existing is not None and existing is not cls:
                raise ValueError(f"Task cfg name '{key}' is already registered to {existing.__name__}.")
            TASK_CFG_REGISTRY[key] = cls
        return cls

    return _decorator


def get_task_cfg_class(name: str):
    """Return the task config class registered under the given name (case-insensitive)."""
    _discover_cfg_modules()
    key = name.strip().lower()
    try:
        return TASK_CFG_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_CFG_REGISTRY.keys())) or "<none>"
        raise KeyError(f"Unknown task cfg '{name}'. Available cfgs: {available}") from exc


def list_task_cfgs():
    """List all registered task cfg names (sorted)."""
    _discover_cfg_modules()
    return sorted(TASK_CFG_REGISTRY.keys())
