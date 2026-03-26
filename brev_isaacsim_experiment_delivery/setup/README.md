# Legacy setup helpers

These wrappers now defer to the repository root launchable:

- `install.sh` calls `scripts/bootstrap_project.sh`
- `launch.sh` starts `../isaac-lab/docker compose up -d`

The active Brev launchable is the repository root `isaac-lab/` directory, not the older placeholder multi-service compose file in this folder.
