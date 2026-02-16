"""Verify trace-based STDP produces positive eligibility for causal pre→post pairing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from biosnn.contracts.learning import LearningBatch
from biosnn.contracts.modulators import ModulatorKind
from biosnn.contracts.neurons import StepContext
from biosnn.learning.rules import RStdpEligibilityParams, RStdpEligibilityRule


def test_trace_causal_pairing() -> None:
    """Pre fires at step 2, post fires at step 5. Eligibility should be positive."""
    params = RStdpEligibilityParams(
        lr=0.1,
        tau_e=0.05,
        a_plus=1.0,
        a_minus=0.6,
        w_min=0.0,
        w_max=1.0,
        weight_decay=0.0,
        dopamine_scale=1.0,
        baseline=0.0,
        tau_pre=0.020,
        tau_post=0.020,
        lazy_decay_enabled=False,
    )
    rule = RStdpEligibilityRule(params)
    state = rule.init_state(4, ctx=StepContext(device="cpu", dtype="float32"))

    assert state.pre_trace is not None, "pre_trace should be initialized"
    assert state.post_trace is not None, "post_trace should be initialized"

    weights = torch.full((4,), 0.5, dtype=torch.float32)
    dt = 0.001

    # Step 0-1: nothing
    for t in range(2):
        pre = torch.zeros(4)
        post = torch.zeros(4)
        batch = LearningBatch(pre_spikes=pre, post_spikes=post, weights=weights)
        state, res = rule.step(state, batch, dt=dt, t=t * dt, ctx=StepContext())

    # Step 2: pre fires (edges 0,1)
    pre = torch.tensor([1.0, 1.0, 0.0, 0.0])
    post = torch.zeros(4)
    batch = LearningBatch(pre_spikes=pre, post_spikes=post, weights=weights)
    state, res = rule.step(state, batch, dt=dt, t=2 * dt, ctx=StepContext())

    print(f"After pre fire: pre_trace = {state.pre_trace.tolist()}")
    print(f"  post_trace = {state.post_trace.tolist()}")
    print(f"  eligibility = {state.eligibility.tolist()}")

    # Step 3-4: nothing
    for t in range(3, 5):
        pre = torch.zeros(4)
        post = torch.zeros(4)
        batch = LearningBatch(pre_spikes=pre, post_spikes=post, weights=weights)
        state, res = rule.step(state, batch, dt=dt, t=t * dt, ctx=StepContext())

    print(f"After decay: pre_trace = {[f'{x:.4f}' for x in state.pre_trace.tolist()]}")
    print(f"  eligibility = {[f'{x:.4f}' for x in state.eligibility.tolist()]}")

    # Step 5: post fires (edges 0,1)
    pre = torch.zeros(4)
    post = torch.tensor([1.0, 1.0, 0.0, 0.0])
    batch = LearningBatch(pre_spikes=pre, post_spikes=post, weights=weights)
    state, res = rule.step(state, batch, dt=dt, t=5 * dt, ctx=StepContext())

    print("After post fire (causal pairing):")
    print(f"  pre_trace = {[f'{x:.4f}' for x in state.pre_trace.tolist()]}")
    print(f"  eligibility = {[f'{x:.6f}' for x in state.eligibility.tolist()]}")

    elig_causal = state.eligibility[0].item()
    elig_noncausal = state.eligibility[2].item()
    print(f"  causal edge elig = {elig_causal:.6f} (should be > 0)")
    print(f"  non-causal edge elig = {elig_noncausal:.6f} (should be ~0)")

    assert elig_causal > 0, f"Causal eligibility should be positive, got {elig_causal}"
    assert abs(elig_noncausal) < 1e-10, f"Non-causal eligibility should be ~0, got {elig_noncausal}"

    # Now apply dopamine and check dw > 0
    pre = torch.zeros(4)
    post = torch.zeros(4)
    dopamine = torch.full((4,), 1.0)
    batch = LearningBatch(
        pre_spikes=pre,
        post_spikes=post,
        weights=weights,
        modulators={ModulatorKind.DOPAMINE: dopamine},
    )
    state, res = rule.step(state, batch, dt=dt, t=6 * dt, ctx=StepContext())

    dw_causal = res.d_weights[0].item()
    dw_noncausal = res.d_weights[2].item()
    print("\nWith DA=1.0:")
    print(f"  dw causal = {dw_causal:.6f} (should be > 0)")
    print(f"  dw non-causal = {dw_noncausal:.6f} (should be ~0)")
    assert dw_causal > 0, f"Weight change for causal edge should be positive, got {dw_causal}"

    print("\n=== TRACE STDP VERIFICATION PASSED ===")


def test_trace_vs_instantaneous() -> None:
    """Compare trace-based vs instantaneous STDP for same-step coincidence."""
    dt = 0.001
    weights = torch.full((4,), 0.5, dtype=torch.float32)

    # Instantaneous (tau_pre=0, tau_post=0)
    params_inst = RStdpEligibilityParams(
        lr=0.1,
        tau_e=0.05,
        a_plus=1.0,
        a_minus=0.6,
        w_min=0.0,
        w_max=1.0,
        weight_decay=0.0,
        tau_pre=0.0,
        tau_post=0.0,
        lazy_decay_enabled=False,
    )
    rule_inst = RStdpEligibilityRule(params_inst)
    state_inst = rule_inst.init_state(4, ctx=StepContext(device="cpu", dtype="float32"))

    # Trace-based (tau_pre=0.020)
    params_trace = RStdpEligibilityParams(
        lr=0.1,
        tau_e=0.05,
        a_plus=1.0,
        a_minus=0.6,
        w_min=0.0,
        w_max=1.0,
        weight_decay=0.0,
        tau_pre=0.020,
        tau_post=0.020,
        lazy_decay_enabled=False,
    )
    rule_trace = RStdpEligibilityRule(params_trace)
    state_trace = rule_trace.init_state(4, ctx=StepContext(device="cpu", dtype="float32"))

    # Scenario: pre fires at step 2, post fires at step 5
    for t in range(10):
        pre = torch.tensor([1.0, 0.0, 0.0, 0.0]) if t == 2 else torch.zeros(4)
        post = torch.tensor([1.0, 0.0, 0.0, 0.0]) if t == 5 else torch.zeros(4)
        batch = LearningBatch(pre_spikes=pre, post_spikes=post, weights=weights)
        state_inst, _ = rule_inst.step(state_inst, batch, dt=dt, t=t * dt, ctx=StepContext())
        state_trace, _ = rule_trace.step(state_trace, batch, dt=dt, t=t * dt, ctx=StepContext())

    elig_inst = state_inst.eligibility[0].item()
    elig_trace = state_trace.eligibility[0].item()

    print("Edge 0 (pre@2, post@5):")
    print(f"  Instantaneous elig: {elig_inst:.6f}")
    print(f"  Trace-based elig:   {elig_trace:.6f}")
    print(f"  Ratio: {elig_trace / elig_inst if abs(elig_inst) > 1e-10 else 'N/A'}")

    if abs(elig_inst) < 1e-10 and abs(elig_trace) > 1e-6:
        print("  → Trace STDP captured the causal pairing that instantaneous MISSED")
    elif abs(elig_trace) > abs(elig_inst):
        print("  → Trace STDP has stronger signal")
    else:
        print("  → WARNING: Trace STDP not better than instantaneous!")


if __name__ == "__main__":
    test_trace_causal_pairing()
    print("\n" + "=" * 60 + "\n")
    test_trace_vs_instantaneous()
