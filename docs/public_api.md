# Public API (biosnn.api)

**Policy:** Only symbols re-exported from `biosnn.api` are considered public and semver-stable.

This log is the canonical list of the current stable façade.

## Current exports

### Version
- `__version__`

### Common
- `StepContext`

### Neurons
- `Compartment`
- `NeuronInputs`
- `NeuronStepResult`
- `INeuronModel`
- `NeuronModelBase`
- `StateTensorSpec`
- `TemplateNeuronModel`
- `GLIFModel`
- `AdEx2CompModel`

### Synapses
- `SynapseTopology`
- `SynapseInputs`
- `SynapseStepResult`
- `ISynapseModel`
- `DelayedCurrentParams`
- `DelayedCurrentSynapse`
- `DelayedSparseMatmulParams`
- `DelayedSparseMatmulSynapse`

### Learning
- `LearningBatch`
- `LearningStepResult`
- `ILearningRule`
- `ThreeFactorHebbianParams`
- `ThreeFactorHebbianRule`

### Neuromodulators
- `ModulatorKind`
- `ModulatorRelease`
- `IModulatorField`

### Monitoring
- `StepEvent`
- `IMonitor`

### Builders
- `NetworkBuilder`
- `NetworkSpec`
- `ErdosRenyi`
- `Init`

### Training
- `Trainer`
- `EngineConfig`
- `TrainReport`

### Presets
- `presets`

## Notes
- Neuron-model specific DTOs like `GLIFParams` and `AdEx2CompParams` exist under
  `biosnn.contracts.neurons` but are **not yet** exported from `biosnn.api` until
  sign conventions and units are finalized.
