from __future__ import annotations

from typing import Type

from .base import BaseDatasetHandler
from .openvla import OpenVLADatasetHandler
from .pi0_libero import Pi0LiberoDatasetHandler
from .open_pi0_simplerenv import OpenPi0SimplerDatasetHandler
from .pi0fast_libero import Pi0FastLiberoDatasetHandler

_REGISTRY: dict[str, Type[BaseDatasetHandler]] = {
    "openvla": OpenVLADatasetHandler,
    "pi0-libero": Pi0LiberoDatasetHandler,
    "open_pi0_simpler_bridge": OpenPi0SimplerDatasetHandler,
    "pi0fast-libero": Pi0FastLiberoDatasetHandler,
}


def get_dataset_handler(name: str | None) -> Type[BaseDatasetHandler]:
    if name is None:
        return BaseDatasetHandler
    return _REGISTRY.get(name, BaseDatasetHandler)

