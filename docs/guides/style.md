# Style guide

- Packages/modules: `snake_case`
- Classes: `PascalCase`
- Keep `biosnn.api` thin and stable.
- Prefer explicit dependencies (constructor injection) over global singletons.
- Put implementation details in internal packages; avoid leaking them through the façade.
