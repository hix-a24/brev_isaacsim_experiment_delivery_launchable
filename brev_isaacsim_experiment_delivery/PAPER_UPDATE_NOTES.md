# Notes on Updating the Manuscript for Brev Compatibility

The original paper was written with a specific software stack and experimental protocol in mind.  When porting the experiments to NVIDIA Brev and Isaac Sim 5.x, several inconsistencies and incompatibilities were discovered.  The points below summarise all deviations from the manuscript and propose concrete changes to the paper so that it accurately reflects what is implementable on Brev.

## Software and Version Mismatches

1. **Python 3.8 vs. Python 3.11** – The paper lists *Python 3.8* in the reproducibility checklist.  However, Isaac Sim 5.x is only compatible with *Python 3.11*【610528769204527†L425-L430】.  For Brev, we therefore use Python 3.11 throughout.  *Manuscript edit*: replace all references to Python 3.8 with Python 3.11 and update installation instructions accordingly.

2. **ROS 2 Foxy vs. ROS 2 Humble** – The paper cites *ROS 2 Foxy* as the middleware.  Isaac Sim’s ROS bridge supports *ROS 2 Humble* and *ROS 2 Jazzy* but not Foxy【860636887936188†L436-L460】.  We choose Humble because it matches Ubuntu 22.04.  *Edit*: update the Methods and Reproducibility sections to specify ROS 2 Humble and remove mention of Foxy.

3. **GPU Requirements** – The paper lists A100 or H100 GPUs in the appendix.  Isaac Sim does not support GPUs without RT cores; A100 and H100 are therefore not supported for real‑time rendering.  Brev provides L40S GPUs, which are supported.  *Edit*: remove A100/H100 from hardware recommendations and instead specify RTX‑class GPUs, such as L40S, as required.

4. **Domain shift severity levels** – The paper defines severity levels 0, 1 and 2 for each domain shift axis【725427445771147†L630-L665】.  To align with typical robustness plots (including our Figure 7 mock‑up), we standardise on *four* levels: 0 (training), 1 (mild), 2 (moderate) and 3 (extreme/combined).  *Edit*: update the text and figure captions to reflect this four‑level scale.

5. **Demonstration counts** – The methods section states 50 demonstrations per task【725427445771147†L564-L566】, whereas the reproducibility checklist specifies 50 demonstrations for Task A and 70 for Task B【725427445771147†L1230-L1267】.  We adopt the 50/70 split from the checklist, as it reflects the actual training protocol.  *Edit*: unify the demonstration count across the manuscript.

6. **Uncertainty sampling** – The paper mentions drawing 10 Monte Carlo dropout samples for uncertainty estimation in one section, yet Table II reports using 20 samples.  We standardise on 20 samples for Monte Carlo dropout.  *Edit*: align the text and table to specify 20 samples.

7. **Confidence interval computation** – The main text claims to use Clopper–Pearson intervals, while the reproducibility table lists Wilson score intervals.  We choose Wilson score intervals for all plotted confidence bands to match the broader robotics literature.  *Edit*: remove mention of Clopper–Pearson and replace with Wilson score.

8. **Figure numbering and content** – Figures 6–10 in the paper are placeholders or schematic diagrams rather than real plots.  In our implementation we will produce data‑driven figures directly from logged simulator outputs.  *Edit*: update figure captions to indicate that the plots are generated from logged results and explain any pilot vs. full‑scale differences.

9. **Sim‑to‑real demonstration (Fig. 11)** – The paper includes a sim‑to‑real experiment (Fig. 11) using a real Franka Panda.  Brev does not provide access to physical robots.  Our pipeline can produce the simulated half of Fig. 11 but not the physical half.  *Edit*: note that sim‑to‑real transfer requires future hardware experiments and cannot be reproduced on Brev alone.

10. **Time budgets and seeds** – Section V mentions running 100 episodes per condition and 3 random seeds.  When budget is constrained (e.g. on a $10 Brev credit), we instead recommend a pilot run with fewer episodes (e.g. 5–20) to validate the pipeline end‑to‑end before scaling up.  *Edit*: clarify that the large‑scale experiment is time and compute intensive and that pilot runs are acceptable for reproduction.

## Other Clarifications

* Document the use of JupyterLab and headless simulation as the primary means of interacting with the experiment.  Remove references to native desktop streaming clients from the text.
* Emphasise that the gating thresholds δ_low=0.2 and δ_high=0.5 come from a calibration process on a validation set【725427445771147†L470-L490】.  Include a brief description of how to tune these thresholds in practice.
* Provide explicit logging schema definitions in the appendix so that other researchers can reproduce Figures 6–10.  Our `data_logs/` folder contains a schema definition file for this purpose.

These notes should be consulted when revising the manuscript to ensure that it aligns with the Brev implementation.  Without these updates, readers may be misled by outdated version numbers or contradictory experiment settings.