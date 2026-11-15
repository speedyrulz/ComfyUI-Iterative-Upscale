# Workflow compatibility

- Existing widget names and order must not change. This is a fundamental project
  rule and applies to fixes, refactoring, and merged changes.
- Preserve the mapping between each widget and its serialized `widgets_values`
  entry. Do not rename, reorder, or insert widgets in a way that shifts existing
  saved values to different controls.
- Fix behavior without changing the existing widget layout or names. Review
  incoming merges for the same compatibility requirement.
- Relevant regression tests must check widget names, order, and saved values
  before and after loading a native workflow.
- Node IDs (the keys of `NODE_CLASS_MAPPINGS`) must not change either; they are
  what saved workflows refer to. Every ID is the Impact Pack ID with a `RulZ`
  prefix (display names: `RulZ ` plus the Impact Pack name), so the pack can be
  installed next to Impact Pack. New nodes follow the same pattern.
- The progress event is `rulz-iterative-upscale/update_status` and the menu
  category root is `RulZ/`; do not switch either back to Impact Pack's names.

# Tests

- `python -m pytest tests` runs offline on plain Python with torch: `tests/conftest.py`
  stubs the ComfyUI modules and loads the pack. No ComfyUI install or GPU is needed.
- `tests/test_widgets.py` pins every node ID and the names and order of its inputs.
  Update it only together with a deliberate, documented compatibility change.
- Behaviour that depends on ComfyUI internals still needs a check against a real
  ComfyUI install.

# Scope

- This pack holds only the Iterative Upscale nodes from ComfyUI Impact Pack and
  the nodes needed to feed them (UPSCALER providers, PK_HOOK providers, KSAMPLER
  providers, the GITS scheduler function and the BASIC_PIPE nodes).
- Keep it free of extra Python dependencies beyond ComfyUI itself.
