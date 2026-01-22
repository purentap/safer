from __future__ import annotations

from typing import Type

from .base import BaseDatasetHandler
from .openvla import OpenVLADatasetHandler
from .pi0_libero import Pi0LiberoDatasetHandler


_REGISTRY: dict[str, Type[BaseDatasetHandler]] = {
    "openvla": OpenVLADatasetHandler,
    "pi0-libero": Pi0LiberoDatasetHandler,
}


def get_dataset_handler(name: str | None) -> Type[BaseDatasetHandler]:
    if name is None:
        return BaseDatasetHandler
    return _REGISTRY.get(name, BaseDatasetHandler)

