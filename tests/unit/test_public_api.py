import pytest
pytestmark = pytest.mark.unit

def test_public_api_exports():
    import biosnn.api as api

    # Core
    assert hasattr(api, "StepContext")

    # Neurons
    assert hasattr(api, "INeuronModel")
    assert hasattr(api, "NeuronInputs")
    assert hasattr(api, "NeuronStepResult")
    assert hasattr(api, "Compartment")

    # Synapses
    assert hasattr(api, "ISynapseModel")
    assert hasattr(api, "SynapseTopology")

    # Learning
    assert hasattr(api, "ILearningRule")
    assert hasattr(api, "LearningBatch")

    # Neuromodulators
    assert hasattr(api, "IModulatorField")
    assert hasattr(api, "ModulatorKind")

    # Monitors
    assert hasattr(api, "IMonitor")
    assert hasattr(api, "StepEvent")
