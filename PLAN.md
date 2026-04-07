# Sim-to-Real Robot Arm — SO-100 Domain Randomization Study

## Project Overview
Train a robotic arm policy entirely in simulation (IsaacGym / Genesis) using RL with domain randomization, then deploy on a real SO-100 arm with zero additional training. Measure how well the simulation policy transfers and use domain randomization to close the sim-to-real gap.

## Research Hypothesis
*Does randomizing physics parameters in simulation produce policies that transfer better to the real robot than a baseline trained without randomization?*

---

## Progress Tracker

<!-- UPDATE THIS FILE AFTER EACH STEP IS COMPLETED -->

### Phase 0: Project Setup [IN PROGRESS]
- [x] Create project folder structure
- [ ] Write initial PLAN.md (this file)
- [ ] Initialize git repository
- [ ] Create requirements.txt with all dependencies
- [ ] Set up base Dockerfile for cloud training

### Phase 1: Simulation Environment
- [ ] Create IsaacGym/Genesis environment definition for SO-100
- [ ] Implement observation space (21-dim vector)
- [ ] Implement action space (6 continuous joint targets)
- [ ] Implement hand-crafted reward function
- [ ] Verify environment in simulation viewer

### Phase 2: Baseline Training
- [ ] Train baseline policy (no randomization) for reach task
- [ ] Extend to grasp task with gripper, object, lift reward
- [ ] Log experiments to Weights & Biases
- [ ] Evaluate baseline policy in simulation (100 episodes)

### Phase 3: Domain Randomization
- [ ] Build domain randomization module
  - Object mass randomization (0.05–0.3 kg)
  - Table friction randomization (0.3–1.2)
  - Joint damping randomization (0.8–1.2x)
  - Camera position noise (-2 to +2 cm)
  - Lighting intensity randomization (0.5–1.5x)
  - Visual randomization (random object colors)
- [ ] Verify parameters sample correctly per episode
- [ ] Train domain randomized policy
- [ ] Compare learning curves vs baseline in W&B
- [ ] Evaluate randomized policy in simulation (100 episodes)

### Phase 4: Real Robot Setup
- [ ] SO-100 arm hardware setup and SDK installation
- [ ] Verify joint control from Python
- [ ] Camera calibration (pixel to world coordinates)
- [ ] YOLOv8 object detection for colored blocks
- [ ] End-to-end real robot inference pipeline

### Phase 5: Real-World Evaluation
- [ ] Deploy baseline policy on real arm (20 grasp attempts)
- [ ] Deploy domain randomized policy on real arm (20 grasp attempts)
- [ ] Run ablation study (low/medium/high randomization)
- [ ] Record all attempts on video

### Phase 6: Analysis & Reporting
- [ ] Plot sim vs real gap results
- [ ] Generate ablation study charts
- [ ] Write final report
- [ ] Create demo video
- [ ] Clean up repository for submission

---

## Architecture
See SYSTEM_ARCHITECTURE.md for full diagrams.

## Timeline
See TIMELINE.md for week-by-week breakdown.

## Key Files
- `sim/envs/so100_env.py` - Simulation environment
- `sim/envs/domain_rand.py` - Domain randomization logic
- `sim/envs/reward.py` - Reward functions
- `sim/train.py` - Main training script
- `real/inference.py` - Real robot inference
- `real/evaluate_real.py` - Real-world evaluation
- `analysis/plot_results.py` - Results visualization
