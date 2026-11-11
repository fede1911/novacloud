# 0004 · Composition root and toolkits

**Context.** Agents, tools and services depend on each other. If every class reads the environment
and creates its own dependencies, it becomes hard to test and to see what it uses.

**Decision.**
- A single composition root, `agent/runtime.py`; the environment is read only in `Settings` and
  `AgentConfig`.
- Every tool is a `build()` function that receives the services (`ToolServices`) and returns a
  `ToolSpec`.
- Products declare their tools in `config/scopes.json`; `PRODUCT_TOOL_BUILDERS` is the explicit list
  of the available tools.

**Rejected alternatives.** Automatic registration with a decorator at import time: fewer lines, but
the list of tools depends on which modules were imported and is not visible in one place.

**Consequences.** Adding a tool means touching four places: its module, `PRODUCT_TOOL_BUILDERS`,
`ProductToolName` and `config/scopes.json`. Tests inject
test doubles without global patches.
