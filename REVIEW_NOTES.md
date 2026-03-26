# Review Notes

The project has been upgraded from a launchable scaffold plus smoke test into a runnable synthetic experiment stack.

## Added or changed

- synthetic task backends for `DrawerObjectTask` and `ClutterSortTask`
- learned lightweight checkpoint generation for OpenVLA-style and Octo-style policies
- executable end-to-end runner at `scripts/run_experiment.py`
- convenience launcher at `scripts/run_end_to_end.sh`
- smoke test wrapper that calls the same pipeline with reduced settings
- robust figure rendering from generated CSV logs
- run manifest emission in `runs/latest/`

## Verified locally

- `python scripts/run_smoke_test.py`
- `./scripts/run_end_to_end.sh`

Both commands complete successfully and produce logs, checkpoints, figures, and a manifest in the expected directories.
