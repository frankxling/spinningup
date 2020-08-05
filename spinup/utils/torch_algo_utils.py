from typing import Union, Callable
import torch.optim


def update_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """
    Update the learning rate for a given optimizer.
    Useful when doing linear schedule.

    :param optimizer: (th.optim.Optimizer)
    :param learning_rate: (float)
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def get_schedule_fn(value_schedule: Union[Callable, float]) -> Callable:
    """
    Transform (if needed) learning rate and clip range (for PPO)
    to callable.

    :param value_schedule: (callable or float)
    :return: (function)
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = ConstCallable(float(value_schedule))
    else:
        assert callable(value_schedule)
    return value_schedule


class ConstCallable:
    def __init__(self, const_val: float):
        self.const_val = const_val

    def __call__(self, timestep):
        return self.const_val
