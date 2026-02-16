#!/usr/bin/env python
"""Debug script to trace dopamine through the learning pipeline."""

import sys

sys.path.insert(0, "src")


import torch

from biosnn.contracts.learning import LearningBatch
from biosnn.contracts.modulators import ModulatorKind
from biosnn.learning.rules.modulated_wrapper import ModulatedRuleWrapper, ModulatedRuleWrapperParams
from biosnn.learning.rules.rstdp_eligibility import RStdpEligibilityParams, RStdpEligibilityRule


def test_learning_with_dopamine():
    """Test if dopamine reaches the learning rule and produces weight changes."""
    print("=" * 80)
    print("TEST 1: RSTDP with dopamine (NO WRAPPER)")
    print("=" * 80)

    # Create simple pre/post spike patterns
    batch_size = 10
    pre_spikes = torch.rand((batch_size,), dtype=torch.float32) > 0.7
    post_spikes = torch.rand((batch_size,), dtype=torch.float32) > 0.7
    weights = torch.rand((batch_size,), dtype=torch.float32) * 0.5 + 0.02
    dopamine = torch.ones((batch_size,), dtype=torch.float32) * 0.5  # <-- KEY: dopamine

    print(f"Pre spikes: {pre_spikes.sum().item()} / {batch_size}")
    print(f"Post spikes: {post_spikes.sum().item()} / {batch_size}")
    print(f"Dopamine level: {dopamine.mean().item():.4f}")
    print(
        f"Weights before: min={weights.min().item():.6f}, max={weights.max().item():.6f}, mean={weights.mean().item():.6f}"
    )

    # Test raw RSTDP
    rule = RStdpEligibilityRule(
        RStdpEligibilityParams(
            tau_e=0.050,
            a_plus=1.0,
            a_minus=1.0,
            lr=0.001,
        )
    )

    state = rule.init_state(e=batch_size, ctx=None)
    batch = LearningBatch(
        pre_spikes=pre_spikes.float(),
        post_spikes=post_spikes.float(),
        weights=weights.clone(),
        modulators={ModulatorKind.DOPAMINE: dopamine},
    )

    new_state, result = rule.step(state, batch, dt=0.001, t=0.0, ctx=None)
    dw = result.d_weights

    print(
        f"dW from RSTDP: min={dw.min().item():.8f}, max={dw.max().item():.8f}, mean={dw.mean().item():.8f}"
    )
    print(f"dW abs sum: {dw.abs().sum().item():.8f}")

    if dw.abs().sum().item() < 1e-9:
        print("❌ ERROR: RSTDP produced zero weight changes with dopamine present!")
        return False
    else:
        print("✅ RSTDP is learning with dopamine")

    print("\n" + "=" * 80)
    print("TEST 2: WRAPPED RSTDP with wrapper gains (0.5, 0.3, 0.03)")
    print("=" * 80)

    # Now wrap it with the problematic wrapper params from failed run
    base_rule = RStdpEligibilityRule(
        RStdpEligibilityParams(
            tau_e=0.050,
            a_plus=1.0,
            a_minus=1.0,
            lr=0.001,
        )
    )

    wrapper_params = ModulatedRuleWrapperParams(
        ach_lr_gain=0.5,  # increased from 0.4
        ne_lr_gain=0.3,  # increased from 0.25
        ht_extra_weight_decay=0.03,  # increased from 0.02
        lr_clip_min=0.1,
        lr_clip_max=10.0,
        combine_mode="exp",
    )

    wrapped_rule = ModulatedRuleWrapper(base_rule, params=wrapper_params)

    # Test with NO modulators first (the failed run case)
    print("\nTest 2a: Wrapper with ZERO modulators (should produce zero dW)")
    wrapped_state = wrapped_rule.init_state(e=batch_size, ctx=None)
    batch_no_mod = LearningBatch(
        pre_spikes=pre_spikes.float(),
        post_spikes=post_spikes.float(),
        weights=weights.clone(),
        modulators={},  # <-- Empty!
    )

    _, result = wrapped_rule.step(wrapped_state, batch_no_mod, dt=0.001, t=0.0, ctx=None)
    dw_no_mod = result.d_weights
    print(
        f"dW with no modulators: min={dw_no_mod.min().item():.8f}, max={dw_no_mod.max().item():.8f}, sum={dw_no_mod.abs().sum().item():.8f}"
    )
    if result.extras is not None:
        print(f"mean_lr_scale: {result.extras['mean_lr_scale'].item():.4f}")

    if dw_no_mod.abs().sum().item() > 1e-9:
        print(
            "⚠️  WARNING: Getting weight changes even with empty modulators (lr_scale must be > 1)"
        )

    # Test with dopamine
    print("\nTest 2b: Wrapper WITH dopamine (should learn)")
    wrapped_state = wrapped_rule.init_state(e=batch_size, ctx=None)
    batch_with_dop = LearningBatch(
        pre_spikes=pre_spikes.float(),
        post_spikes=post_spikes.float(),
        weights=weights.clone(),
        modulators={ModulatorKind.DOPAMINE: dopamine},
    )

    _, result = wrapped_rule.step(wrapped_state, batch_with_dop, dt=0.001, t=0.0, ctx=None)
    dw_with_mod = result.d_weights
    print(
        f"dW with dopamine: min={dw_with_mod.min().item():.8f}, max={dw_with_mod.max().item():.8f}, sum={dw_with_mod.abs().sum().item():.8f}"
    )
    if result.extras is not None:
        print(f"mean_lr_scale: {result.extras['mean_lr_scale'].item():.4f}")

    if dw_with_mod.abs().sum().item() < 1e-9:
        print("❌ ERROR: Wrapper produced zero weight changes even WITH dopamine!")
        print("   This is the bug!")
        return False

    print("\n" + "=" * 80)
    print("TEST 3: Check if wrapper gains are causing lr_scale explosion")
    print("=" * 80)

    # Check what the lr_scale would be with different ACH levels
    for ach_level in [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]:
        mod_arg = wrapper_params.ach_lr_gain * (ach_level - wrapper_params.ach_baseline)
        lr_scale_exp = (
            mod_arg.exp() if isinstance(mod_arg, torch.Tensor) else torch.tensor(mod_arg).exp()
        )
        lr_scale_clipped = torch.clamp(
            lr_scale_exp, min=wrapper_params.lr_clip_min, max=wrapper_params.lr_clip_max
        )
        print(
            f"  ACH={ach_level:.1f}: mod_arg={float(mod_arg):.4f}, exp={float(lr_scale_exp):.4f}, clipped={float(lr_scale_clipped):.4f}"
        )


if __name__ == "__main__":
    try:
        success = test_learning_with_dopamine()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ EXCEPTION: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
