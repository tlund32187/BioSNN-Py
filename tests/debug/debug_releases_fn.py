#!/usr/bin/env python3
"""Debug script to test if releases_fn is being called and returning dopamine."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import torch  # noqa: E402

from biosnn.contracts.modulators import ModulatorKind, ModulatorRelease  # noqa: E402


def test_releases_fn_integration():
    """Test if releases_fn can be called and returns dopamine."""
    print("=" * 80)
    print("TEST: Releases_fn integration")
    print("=" * 80)

    # Simulated pending_releases dict (like what engine_runner has)
    pending_releases = {
        ModulatorKind.DOPAMINE: 0.5,  # Some dopamine is pending
    }
    modulator_kinds = (ModulatorKind.DOPAMINE,)

    # Simulate what engine_runner's releases_fn does
    def mock_releases_fn():
        """Simplified version of _build_modulator_releases_for_step."""
        releases = []
        for kind in modulator_kinds:
            amount = pending_releases.get(kind, 0.0)
            if amount != 0.0:
                releases.append(
                    ModulatorRelease(
                        kind=kind,
                        positions=torch.tensor([[0.5, 0.5]], dtype=torch.float32),  # 1 release site
                        amount=torch.tensor([amount], dtype=torch.float32),
                    )
                )
        return releases

    # Test 1: Check if releases_fn produces dopamine
    releases = mock_releases_fn()
    print(f"\nNumber of releases: {len(releases)}")
    for r in releases:
        print(f"  Release: kind={r.kind}, amount={r.amount}, positions={r.positions}")

    dopamine_releases = [r for r in releases if r.kind == ModulatorKind.DOPAMINE]
    total_dopamine = sum(r.amount for r in dopamine_releases)
    print(f"\nTotal dopamine in releases: {total_dopamine}")

    if total_dopamine > 0:
        print("✅ releases_fn is producing dopamine")
    else:
        print("❌ releases_fn NOT producing dopamine (PROBLEM!)")

    # Test 2: After calling releases_fn, check if pending_releases should be cleared
    print("\n" + "-" * 80)
    print("TEST: After releases_fn, should pending_releases be cleared?")
    print("-" * 80)
    print(f"pending_releases before: {pending_releases}")
    # In engine_runner, after _build_modulator_releases_for_step, pending_releases is NOT cleared
    # It's only cleared if reward_delivery_steps <= 0 AND has_pending_dopamine is False
    print(f"pending_releases after: {pending_releases}")
    print("✅ pending_releases still contains dopamine (correct)")

    # Test 3: Simulate what _discard_pending_releases does
    print("\n" + "-" * 80)
    print("TEST: What _discard_pending_releases does")
    print("-" * 80)

    def _discard_pending_releases(pending_releases):
        for kind in modulator_kinds:
            pending_releases[kind] = 0.0

    test_pending = pending_releases.copy()
    print(f"Before discard: {test_pending}")
    _discard_pending_releases(test_pending)
    print(f"After discard: {test_pending}")
    print("✅ Discard correctly zeros out pending_releases")

    # Test 4: Check the actual condition logic for dopamine
    print("\n" + "-" * 80)
    print("TEST: Condition check for has_pending_dopamine")
    print("-" * 80)
    has_pending_dopamine = bool(
        float(pending_releases.get(ModulatorKind.DOPAMINE, 0.0)) != 0.0
    )
    print(f"has_pending_dopamine: {has_pending_dopamine}")
    reward_delivery_steps = 0
    effective_reward_steps = (
        max(1, int(reward_delivery_steps))
        if has_pending_dopamine
        else int(reward_delivery_steps)
    )
    print(f"reward_delivery_steps={reward_delivery_steps}")
    print(f"effective_reward_steps={effective_reward_steps}")
    if effective_reward_steps > 0:
        print("✅ Will execute reward delivery steps (dopamine will be processed)")
    else:
        print("❌ Will NOT execute reward delivery steps (dopamine will be DISCARDED!)")

    # Test 5: Check if dopamine gets reset somewhere else
    print("\n" + "-" * 80)
    print("TEST: Check if dopamine gets reset after being queued")
    print("-" * 80)
    # In _queue_trial_feedback_releases, dopamine should be set
    # But then in _run_trial_steps, it should NOT be reset
    # Let me check the code flow...
    print("Need to check if pending_releases is reset anywhere between queueing and processing")
    print("This would require tracing through _run_trial_steps and releases_fn calls")

if __name__ == "__main__":
    test_releases_fn_integration()
