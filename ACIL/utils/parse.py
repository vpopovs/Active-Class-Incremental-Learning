import inspect


def match_parameters(func: callable, args: dict) -> dict:
    """
    Matches the parameters of a function with the arguments provided.

    Args:
        func (function): Function to match.
        args (dict): Arguments to match.
    Returns:
        dict: Matched parameters.
    """
    kwargs = {}
    parameters = inspect.signature(func).parameters.keys()
    if "args" in parameters and "kwargs" in parameters:
        return func(**args)

    for key in parameters:
        kwargs[key] = args[key]
    return func(**kwargs)
