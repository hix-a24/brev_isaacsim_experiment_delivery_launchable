# Data Logging Schema

The experiment logs all relevant information into structured CSV files within the `data_logs/` directory.  This document defines the schema for each log file.  When adding new fields or logs, update this file accordingly.

## episodes.csv

One row per episode.  Columns:

| Column                    | Type    | Description |
|---------------------------|---------|-------------|
| `run_id`                  | string  | Identifier grouping all episodes belonging to one run configuration (e.g. timestamp or UUID). |
| `task`                    | string  | Name of the task: `drawer` or `clutter_sort`. |
| `method`                  | string  | Safety mechanism used: `baseline` (no gating), `uncalibrated`, `calibrated`, `oracle_fallback`. |
| `policy`                  | string  | Policy model: `openvla` or `octo`. |
| `seed`                    | int     | Random seed for reproducibility. |
| `shift_axis`              | string  | Domain shift axis: `lighting`, `texture`, `occlusion`, `sensor`, `combined`. |
| `shift_severity`          | int     | Shift level: 0 (train domain) to 3 (extreme/combined). |
| `instruction`             | string  | Natural language instruction provided to the policy. |
| `success`                 | bool    | Whether the episode succeeded (all subgoals completed). |
| `failure_reason`          | string  | Reason for failure if `success=false` (e.g. `timeout`, `collision`). |
| `collision_any`           | bool    | True if any collisions occurred during the episode. |
| `collision_count`         | int     | Number of discrete collision events. |
| `peak_contact_force_n`    | float   | Maximum contact force recorded (Newton). |
| `peak_contact_torque_nm`  | float   | Maximum contact torque recorded (Newton‑meter). |
| `num_interventions`       | int     | Total number of supervisor interventions (pause or fallback). |
| `num_pause_reobserve`     | int     | Number of pause/reobserve events. |
| `num_fallbacks`           | int     | Number of fallback events executed. |
| `episode_wall_time_s`     | float   | Real time elapsed for the episode (seconds). |
| `sim_time_s`              | float   | Simulation time elapsed (seconds). |
| `object_in_bin`           | bool    | For Task A: whether the target block ended up in the bin. |
| `drawer_closed`           | bool    | For Task A: whether the drawer was closed at the end. |
| `timed_out`               | bool    | Whether the episode timed out before completion. |
| `video_path`              | string  | Path to the recorded MP4 file for the episode, if videos are saved. |

## steps.csv

One row per time step of every episode.  Columns:

| Column                      | Type    | Description |
|-----------------------------|---------|-------------|
| `episode_id`                | string  | Unique identifier linking back to the episode (e.g. combination of `run_id` and episode index). |
| `t_step`                    | int     | Discrete time step index. |
| `phase`                     | string  | High‑level phase of the task (e.g. `approach_handle`, `grasp`, `place`, `reobserve`, `fallback`). |
| `ee_pose`                   | string  | End‑effector pose encoded as a JSON or semicolon‑delimited string. |
| `joint_positions`           | string  | Joint positions vector encoded similarly. |
| `raw_policy_confidence`     | float   | Mean confidence of the policy over MC/ensemble samples. |
| `raw_uncertainty`           | float   | Another metric capturing uncertainty (e.g. variance of action samples). |
| `uncertainty_entropy`       | float   | Entropy of the confidence distribution. |
| `uncertainty_variance`      | float   | Variance of the action samples. |
| `calibrated_success_prob`   | float   | Calibrated success probability after temperature scaling. |
| `calibrated_failure_risk`   | float   | Risk estimate r_t computed by the risk estimator. |
| `gating_state`              | string  | Current control state: `proceed`, `pause_reobserve`, or `fallback`. |
| `action_vector`             | string  | Action applied at this step (encoded as a list). |
| `observation_frame_path`    | string  | Path to the saved frame image (optional, for debugging). |

## interventions.csv

One row per intervention event.  Columns:

| Column          | Type    | Description |
|-----------------|---------|-------------|
| `episode_id`    | string  | Identifier linking to the episode. |
| `t_step`        | int     | Time step of the intervention. |
| `event_type`    | string  | Type of intervention: `pause`, `fallback`, `safe_stop`. |
| `phase`         | string  | Task phase during which the intervention occurred. |
| `risk_before`   | float   | Risk estimate immediately before the intervention. |
| `risk_after`    | float   | Risk estimate after the intervention (e.g. after re‑observe or fallback). |
| `world_x`       | float   | World x‑coordinate of the end effector at intervention. |
| `world_y`       | float   | World y‑coordinate of the end effector at intervention. |
| `world_z`       | float   | World z‑coordinate of the end effector at intervention. |
| `resolved_by`   | string  | How the intervention was resolved: `pause_reobserve`, `fallback_planner`, `safe_stop`. |
| `resume_success`| bool    | Whether the system successfully resumed nominal execution after the intervention. |

## contacts.csv

One row per contact or proximity event.  Columns:

| Column                  | Type    | Description |
|-------------------------|---------|-------------|
| `episode_id`            | string  | Identifier linking to the episode. |
| `t_step`                | int     | Time step at which contact occurred. |
| `link_name`             | string  | Name of the robot link involved in the contact. |
| `other_object`          | string  | Name of the other object or environment part contacted. |
| `contact_force_n`       | float   | Magnitude of the contact force (Newton). |
| `contact_torque_nm`     | float   | Magnitude of the contact torque (Newton‑meter). |
| `incident_type`         | string  | Category of the contact (e.g. `gentle`, `collision`). |
| `margin_to_collision_m` | float   | Signed distance between the robot’s collision geometry and other objects (positive if separated, zero or negative if intersecting). |

## Notes

* All CSV files are append‑only.  Logs from multiple runs should be
  concatenated by appending rows.
* Consider converting logs to Parquet for faster I/O when running large
  experiments; update `src/logger` accordingly.
* The `success` column in `steps.csv` is optional and may not be available at
  step granularity.  If absent, reliability diagrams will not be computed.
* For the pilot runs, you may leave some fields blank or default them to
  sensible values; the figure generation scripts handle missing data gracefully.