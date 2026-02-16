"""
Debug dopamine spatial distribution in field across both runs.
Check if dopamine is released at output positions but not sampled at hidden/input positions.
"""

import json
from pathlib import Path

import torch

from biosnn.contracts.modulators import ModulatorKind, ModulatorRelease
from biosnn.neuromodulators import GridDiffusion2DField, GridDiffusion2DParams

# Load both run configs
working_run_dir = "artifacts/run_20260212_132912_619800"
broken_run_dir = "artifacts/run_20260212_140714_134900"

configs = {}
for run_name, run_dir in [("WORKING", working_run_dir), ("BROKEN", broken_run_dir)]:
    config_path = Path(run_dir) / "run_config.json"
    with open(config_path) as f:
        configs[run_name] = json.load(f)

# Check field configuration
for run_name, config in configs.items():
    mod_config = config["modulators"]
    print(f"\n{run_name} RUN - Modulator Config:")
    print(f"  field_type: {mod_config['field_type']}")
    print(f"  grid_size: {mod_config['grid_size']}")
    print(f"  world_extent: {mod_config['world_extent']}")
    print(f"  deposit_sigma: {mod_config['deposit_sigma']}")
    print(f"  diffusion: {mod_config['diffusion']}")
    print(f"  decay_tau: {mod_config['decay_tau']}")

# Create a GridDiffusion2DField and test dopamine diffusion
print("\n" + "=" * 70)
print("SPATIAL DISTRIBUTION TEST")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

world_extent = (1.0, 1.0)
grid_size = (16, 16)

field = GridDiffusion2DField(
    params=GridDiffusion2DParams(
        grid_size=grid_size,
        world_extent=world_extent,
        decay_tau=0.05,
        diffusion=0.0,
        deposit_sigma=0.0,
    )
)

# Define three test positions: corners and where dopamine is released
input_positions = torch.tensor(
    [
        [0.2, 0.2, 0.0],  # Bottom-left (input area)
        [0.8, 0.2, 0.0],  # Bottom-right
        [0.2, 0.8, 0.0],  # Top-left
    ],
    device=device,
    dtype=dtype,
)

hidden_positions = torch.tensor(
    [
        [0.3, 0.3, 0.0],
        [0.5, 0.5, 0.0],
        [0.7, 0.7, 0.0],
        [0.5, 0.3, 0.0],
    ],
    device=device,
    dtype=dtype,
)

output_positions = torch.tensor(
    [
        [0.25, 0.75, 0.0],  # Output 0
        [0.75, 0.75, 0.0],  # Output 1
    ],
    device=device,
    dtype=dtype,
)

dopamine_release_position = output_positions[0:1]  # Release at first output neuron

# Initialize state
state = field.init_state(ctx=None)

# Deposit dopamine at one output position
release = ModulatorRelease(
    kind=ModulatorKind.DOPAMINE,
    positions=dopamine_release_position,
    amount=torch.tensor([1.0], device=device, dtype=dtype),
)
state = field.step(state, releases=[release], dt=0.001, t=0.0, ctx=None)

# Sample at different positions
print("\nDopamine samples after depositing at output_position[0] (0.25, 0.75):")
print("\nINPUT positions (bottom area):")
input_samples = field.sample_at(
    state, positions=input_positions, kind=ModulatorKind.DOPAMINE, ctx=None
)
for _, (pos, sample) in enumerate(zip(input_positions, input_samples, strict=False)):
    print(f"  [{pos[0]:.2f}, {pos[1]:.2f}]: {sample.item():.6f}")

print("\nHIDDEN positions (mid area):")
hidden_samples = field.sample_at(
    state, positions=hidden_positions, kind=ModulatorKind.DOPAMINE, ctx=None
)
for _, (pos, sample) in enumerate(zip(hidden_positions, hidden_samples, strict=False)):
    print(f"  [{pos[0]:.2f}, {pos[1]:.2f}]: {sample.item():.6f}")

print("\nOUTPUT positions (top area):")
output_samples = field.sample_at(
    state, positions=output_positions, kind=ModulatorKind.DOPAMINE, ctx=None
)
for _, (pos, sample) in enumerate(zip(output_positions, output_samples, strict=False)):
    print(f"  [{pos[0]:.2f}, {pos[1]:.2f}]: {sample.item():.6f}")

# Check grid values to see if dopamine is actually in the field
print("\n" + "=" * 70)
print("GRID CONCENTRATION CHECK")
print("=" * 70)
print(f"Grid shape: {state.grid.shape}")
print(f"Min concentration: {state.grid.min().item():.6f}")
print(f"Max concentration: {state.grid.max().item():.6f}")
print(f"Mean concentration: {state.grid.mean().item():.6f}")
print(f"Total dopamine (sum): {state.grid.sum().item():.6f}")

# Show grid heatmap (simplified)
grid_2d = state.grid.squeeze().cpu().numpy()
print("\nGrid heatmap (showing values > 0.01):")
for row in range(grid_2d.shape[0]):
    line = ""
    for col in range(grid_2d.shape[1]):
        val = grid_2d[row, col]
        if val > 0.01:
            line += f"{val:5.2f} "
        else:
            line += "  .   "
    if line.strip() != "":  # Only print non-empty rows
        print(f"  {line}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("If dopamine is released at output positions but not sampled at")
print("hidden/input positions, then spatial mismatch is the issue.")
print("If samples are zero everywhere, dopamine isn't being deposited.")
print("Otherwise, dopamine is spatially distributed correctly.")
