# ADR 0001: Library-first architecture

**Status:** Accepted  
**Date:** 2026-02-02

## Context
We want a maintainable, modular codebase with a clean public surface. The project will evolve rapidly
(new neuron models, learning rules, backends), but users should not be forced to track internal refactors.

## Decision
- Treat `biosnn` as a **library**.
- Define a **stable façade** in `biosnn.api` that re-exports only supported public symbols.
- Keep experiments and demos in `biosnn.experiments` (not part of public API).

## Consequences
- Internal packages may be reorganized freely.
- Deprecations are centralized and consistent (single policy).
- Users consume a small surface area, reducing churn.
