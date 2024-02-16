import typing
from typing import Callable, Any, Optional, Protocol, TypedDict, Dict
from enum import Enum

try:
    from jax.typing import ArrayLike
except:
    ArrayLike = Any

from jax_md.energy import NeighborList

# Energy Functions

class EnergyFn(Protocol):
    def __call__(self,
                 position: ArrayLike,
                 neighbor: NeighborList=None,
                 **kwargs) -> ArrayLike: ...


EnergyFnTemplate = Callable[[Any], EnergyFn]


# Quantities

class TrajFn(Protocol):
    def __call__(self, quantity_trajs: Dict[str, ArrayLike], weights: ArrayLike = None) -> ArrayLike: ...


class SingleTarget(TypedDict):
    gamma: ArrayLike
    target: ArrayLike
    traj_fn: TrajFn


class QuantityComputeFunction(Protocol):
    def __call__(self, state: Any, **kwargs) -> ArrayLike: ...


QuantityDict = Dict[str, QuantityComputeFunction]

# TODO: Add quantities as enum
# class TargetKey(Enum):
#     KAPPA = "kappa"
#     ALPHA = "alpha"
#     CP = "cp"
#     VOLUME = "volume"
#     ENERGY = "energy"


TargetDict = Dict[str, SingleTarget]


class ComputeFn(Protocol):

    @typing.overload
    def __call__(self, state, neighbor: NeighborList = None, **kwargs) -> Any: ...
    def __call__(self, state, **kwargs) -> Any: ...
