"""Neuron model registry for biophysics implementations."""

from __future__ import annotations

from typing import Any

from biosnn.biophysics.models.adex_2c import AdEx2CompModel
from biosnn.biophysics.models.glif import GLIFModel
from biosnn.contracts.factories import Registry
from biosnn.contracts.neurons import INeuronModel

NEURON_MODELS = Registry[INeuronModel](label="neuron_models")
NEURON_MODELS.register("glif", GLIFModel)
NEURON_MODELS.register("adex_2c", AdEx2CompModel)
NEURON_MODELS.register_alias("adex2c", "adex_2c", deprecated=True)


def create_neuron_model(key: str, **kwargs: Any) -> INeuronModel:
    return NEURON_MODELS.create(key, **kwargs)


__all__ = ["NEURON_MODELS", "create_neuron_model"]
