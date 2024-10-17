from importlib import import_module


def import_from_cfg(cfg: str) -> callable:
    """
    Imports a class from a configuration string.

    Args:
        cfg (str): Configuration string in the format "module.class".
    Returns:
        callable: Imported class.
    """
    module_name, class_name = cfg.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, class_name)
