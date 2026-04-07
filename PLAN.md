     1|# Sim-to-Real Robot Arm — SO-100 Domain Randomization Study
     2|
     3|## Project Overview
     4|Train a robotic arm policy entirely in simulation (IsaacGym / Genesis) using RL with domain randomization, then deploy on a real SO-100 arm with zero additional training. Measure how well the simulation policy transfers and use domain randomization to close the sim-to-real gap.
     5|
     6|## Research Hypothesis
     7|*Does randomizing physics parameters in simulation produce policies that transfer better to the real robot than a baseline trained without randomization?*
     8|
     9|---
    10|
    11|## Progress Tracker
    12|
    13|<!-- UPDATE THIS FILE AFTER EACH STEP IS COMPLETED -->
    14|
    15|### Phase 0: Project Setup [IN PROGRESS]
    16|- [x] Create project folder structure
    17|- [ ] Write initial PLAN.md (this file)
    18|- [ ] Initialize git repository
    19|- [ ] Create requirements.txt with all dependencies
    20|- [ ] Set up base Dockerfile for cloud training
    21|
    22|### Phase 1: Simulation Environment
    23|- [ ] Create IsaacGym/Genesis environment definition for SO-100
    24|- [ ] Implement observation space (21-dim vector)
    25|- [ ] Implement action space (6 continuous joint targets)
    26|- [ ] Implement hand-crafted reward function
    27|- [ ] Verify environment in simulation viewer
    28|
    29|### Phase 2: Baseline Training
    30|- [ ] Train baseline policy (no randomization) for reach task
    31|- [ ] Extend to grasp task with gripper, object, lift reward
    32|- [ ] Log experiments to Weights & Biases
    33|- [ ] Evaluate baseline policy in simulation (100 episodes)
    34|
    35|### Phase 3: Domain Randomization
    36|- [ ] Build domain randomization module
    37|  - Object mass randomization (0.05–0.3 kg)
    38|  - Table friction randomization (0.3–1.2)
    39|  - Joint damping randomization (0.8–1.2x)
    40|  - Camera position noise (-2 to +2 cm)
    41|  - Lighting intensity randomization (0.5–1.5x)
    42|  - Visual randomization (random object colors)
    43|- [ ] Verify parameters sample correctly per episode
    44|- [ ] Train domain randomized policy
    45|- [ ] Compare learning curves vs baseline in W&B
    46|- [ ] Evaluate randomized policy in simulation (100 episodes)
    47|
    48|### Phase 4: Real Robot Setup
    49|- [ ] SO-100 arm hardware setup and SDK installation
    50|- [ ] Verify joint control from Python
    51|- [ ] Camera calibration (pixel to world coordinates)
    52|- [ ] YOLOv8 object detection for colored blocks
    53|- [ ] End-to-end real robot inference pipeline
    54|
    55|### Phase 5: Real-World Evaluation
    56|- [ ] Deploy baseline policy on real arm (20 grasp attempts)
    57|- [ ] Deploy domain randomized policy on real arm (20 grasp attempts)
    58|- [ ] Run ablation study (low/medium/high randomization)
    59|- [ ] Record all attempts on video
    60|
    61|### Phase 6: Analysis & Reporting
    62|- [ ] Plot sim vs real gap results
    63|- [ ] Generate ablation study charts
    64|- [ ] Write final report
    65|- [ ] Create demo video
    66|- [ ] Clean up repository for submission
    67|
    68|---
    69|
    70|## Architecture
    71|See SYSTEM_ARCHITECTURE.md for full diagrams.
    72|
    73|## Timeline
    74|See TIMELINE.md for week-by-week breakdown.
    75|
    76|## Key Files
    77|- `sim/envs/so100_env.py` - Simulation environment
    78|- `sim/envs/domain_rand.py` - Domain randomization logic
    79|- `sim/envs/reward.py` - Reward functions
    80|- `sim/train.py` - Main training script
    81|- `real/inference.py` - Real robot inference
    82|- `real/evaluate_real.py` - Real-world evaluation
    83|- `analysis/plot_results.py` - Results visualization
    84|