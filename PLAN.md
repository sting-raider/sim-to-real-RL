# Sim-to-Real 6-DOF Robot Arm — Domain Randomization Study

> **Goal:** Train a 6-DOF robotic arm to grasp and manipulate objects *entirely*
> inside a physics simulation using RL + domain randomization, then deploy the
> learned policy onto real hardware with **zero additional training**.  Measure
> and analyse the sim-to-real transfer gap.

---

## Research Hypothesis

*Does randomising physics parameters during simulation training produce
policies that transfer to the real robot more reliably than an identical
policy trained without randomisation?*

Secondary question: *What is the optimal range of randomisation — does
increasing the range indefinitely help, or is there a sweet spot?*

---

## Hardware Selection — 6-DOF Arm

The project must work with **any servo-driven 6-DOF arm** that exposes joint
position control over USB/Serial.  After evaluating available platforms for
RL compatibility, cost, community support, and SDK maturity, the following
ranking emerges:

| Arm | DOF | RL Ecosystem | URDF/Sim Support | Python SDK | Approx. Cost |
|---|---|---|---|---|---|
| **SO-ARM101** (LeRobot) | 6 + gripper | ★★★★★ native LeRobot | ★★★★★ maintained URDF, MJCF | `lerobot` (HuggingFace) | ₹8k–12k (DIY kit) |
| Koch v1.1 (LeRobot) | 5 + gripper | ★★★★☆ | ★★★★☆ community | `lerobot` | ₹7k–10k |
| MyCobot 280 (Elephant) | 6 | ★★★☆☆ | ★★★☆☆ | `pymycobot` | ₹25k–30k |
| AR4 (Annin Robotics) | 6 | ★★☆☆☆ | ★★☆☆☆ manual config | ROS/custom | ₹20k+ |

### Primary Recommendation: **SO-ARM101**

- Designed *for* ML/RL research by HuggingFace.
- Open-source 3D-printed structure → easy to repair / modify.
- STS3215 bus servos (30 kg·cm @ 12 V) — plenty of torque for grasping.
- 400 g payload — sufficient for foam blocks, lightweight household objects.
- Fully supported by the **LeRobot** Python library (datasets, pre-trained
  models, teleoperation, replay).
- Well-maintained URDF files → drop into PyBullet / MuJoCo / IsaacGym.
- Active Discord / GitHub community for troubleshooting.

> **The codebase is written to be arm-agnostic.**  The `real/` inference layer
> communicates through an abstract `RobotArm` interface.  Swapping to a
> different 6-DOF arm means implementing a thin adapter — no RL or sim code
> changes.

---

## Simulator Strategy

| Simulator | Role | Why |
|---|---|---|
| **PyBullet** | Development & unit testing (local) | Already integrated; free; CPU-only OK for small runs |
| **MuJoCo** *(stretch)* | Higher-fidelity local sim | Better contact physics; Gymnasium-native; free |
| **IsaacGym / Isaac Sim** *(cloud)* | Massively-parallel GPU training | 512+ envs on one GPU; 10–100× faster than PyBullet |

The environment (`sim/envs/`) is structured so the **observation space, action
space, reward, and domain randomization** are simulator-agnostic.  Only the
physics backend gets swapped.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    CLOUD  (Training Only)                     │
│                                                              │
│  ┌────────────────┐     ┌───────────────────────┐            │
│  │  Simulated     │────▶│  RL Agent (SAC / PPO) │            │
│  │  6-DOF Arm     │◀────│  Policy Network (MLP) │            │
│  │  + graspable   │     └──────────┬────────────┘            │
│  │    objects      │               │                         │
│  └───────┬────────┘         policy.zip saved                 │
│          │                         │                         │
│  Domain Randomization        Weights & Biases                │
│  • friction  • mass          (learning curves,               │
│  • damping   • noise          hyperparams)                   │
│  • colours   • action lag                                    │
└──────────────────────────────┬───────────────────────────────┘
                               │  download policy.zip
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                YOUR LAPTOP  (Inference Only)                  │
│                                                              │
│  ┌────────┐    ┌───────────┐    ┌─────────────┐             │
│  │ USB    │───▶│  YOLOv8   │───▶│  policy.zip │             │
│  │ Webcam │    │  object   │    │  (PyTorch)  │             │
│  └────────┘    │  detect   │    └──────┬──────┘             │
│                └───────────┘           │                     │
│                                  joint commands              │
│                                        ▼                     │
│                                ┌──────────────┐             │
│                                │  Real 6-DOF  │             │
│                                │  Arm (USB)   │             │
│                                └──────────────┘             │
└──────────────────────────────────────────────────────────────┘
```

---

## Observation / Action / Reward Design

### Observation Space — 21 dimensions

| Index | Name | Dims |
|---|---|---|
| 0–5 | Joint positions (radians) | 6 |
| 6–11 | Joint velocities (rad/s) | 6 |
| 12–14 | End-effector position (m) | 3 |
| 15–17 | Object position (m) | 3 |
| 18–20 | Object-to-EE vector (m) | 3 |

### Action Space — 6 continuous values in `[-1, 1]`

Each value maps linearly to the corresponding joint's angular range.
Joint 6 (last axis) doubles as gripper open/close control:
`-1 → fully open`, `+1 → fully closed`.

### Reward (hand-crafted, task-dependent)

```
R(s) = −α · d(ee, obj)                       # approach shaping
     + β · 𝟙[grasping]                       # grasp bonus
     + γ · max(0, h_obj − h_table)           # lift bonus
     − δ · ‖a‖²                              # action penalty
     − ε                                      # time penalty
```

Weights `α, β, γ, δ, ε` are stored in `sim/envs/reward.py:RewardConfig`.

### Domain Randomization Parameters

| Parameter | None | Low | Medium | High | Extreme |
|---|---|---|---|---|---|
| Object mass (kg) | 0.10 | 0.08–0.12 | 0.05–0.30 | 0.02–0.50 | 0.01–1.00 |
| Table friction | 0.80 | 0.70–0.90 | 0.30–1.20 | 0.10–2.00 | 0.00–3.00 |
| Joint damping (×) | 1.00 | 0.95–1.05 | 0.80–1.20 | 0.50–1.50 | 0.30–2.00 |
| Lighting (×) | 1.00 | 0.90–1.10 | 0.50–1.50 | 0.30–2.00 | 0.10–3.00 |
| Camera noise (m) | 0.00 | ±0.005 | ±0.020 | ±0.050 | ±0.100 |
| Action noise (σ) | 0.00 | 0.01 | 0.02 | 0.05 | 0.10 |
| Object colour | fixed | random | random | random | random |

Five preset levels are already implemented in `sim/envs/domain_rand.py`.

---

## The Three Experiments

| # | Name | Training Config | Real-World Eval | Purpose |
|---|---|---|---|---|
| 1 | **Baseline** | `randomization=none` | 20 grasp attempts | Control group — no randomisation |
| 2 | **Domain-Rand** | `randomization=medium` | 20 grasp attempts | Test if randomisation closes gap |
| 3 | **Ablation** | `none / low / medium / high` | 20 grasps × 4 | Find the sweet-spot of randomisation range |

### Expected Outcome Shape

| Policy | Sim Success | Real Success | Gap |
|---|---|---|---|
| Baseline (none) | ~94 % | ~25–35 % | **~60 %** |
| Domain-Rand (medium) | ~85 % | ~60–70 % | **~18 %** |
| Heavy-Rand (high) | ~70 % | ~55–65 % | **~10 %** |

The story: randomisation *trades sim performance for real-world robustness*.

---

## Hardware Bill of Materials

### 🦾 Robot Arm
| Item | Purpose | Est. Cost |
|---|---|---|
| SO-ARM101 kit (or locally sourced BOM) | Real 6-DOF arm | ₹8,000–12,000 |
| STS3215 bus servos × 6 (if sourcing parts) | Actuators | (incl. above) |
| USB Serial bus adapter | Laptop ↔ arm | (incl. above) |
| Sturdy table clamp / base plate | Fix arm to table | ₹300–500 |

### 📷 Vision
| Item | Purpose | Est. Cost |
|---|---|---|
| Logitech C270 (or similar) USB webcam | Top-down workspace view | ₹1,500 |
| Flexible gooseneck clamp mount | Position camera above table | ₹400 |
| Coloured foam cubes (3–4 pcs) | Grasping targets | ₹200 |
| Contrasting table mat (60 × 60 cm) | Clean background for detection | ₹150 |

### 🏗️ Workspace
| Item | Purpose | Est. Cost |
|---|---|---|
| Plywood board 60 × 60 cm | Defined workspace boundary | ₹300 |
| Painter's tape | Mark zones | ₹100 |
| Measuring tape / ruler | Calibrate real vs sim dims | ₹100 |

### ☁️ Cloud Compute
| Item | Purpose | Est. Cost |
|---|---|---|
| RunPod / vast.ai — RTX 3090 / A100 | GPU training (30–50 hrs total) | ₹2,000–4,000 |

### 💰 Total Budget

| Category | Cost |
|---|---|
| Arm hardware | ₹10,000 |
| Vision | ₹1,900 |
| Workspace | ₹500 |
| Cloud GPU | ₹3,000 |
| Buffer | ₹600 |
| **Total** | **~₹16,000** |

---

## Project Folder Structure

```
sim-to-real-arm/
├── sim/
│   ├── envs/
│   │   ├── so100_env.py              ← PyBullet env (6-DOF arm)
│   │   ├── domain_rand.py            ← Randomisation configs & logic
│   │   └── reward.py                 ← Reward functions + adaptive reward
│   ├── configs/                      ← YAML training configurations
│   │   ├── baseline.yaml
│   │   ├── domain_rand.yaml
│   │   └── ablation.yaml
│   ├── train.py                      ← Main training script (SAC/PPO)
│   └── evaluate_sim.py               ← Sim benchmark (N episodes)
│
├── real/
│   ├── inference.py                  ← Run policy on real arm
│   ├── calibrate_camera.py           ← Pixel → world coords tool
│   ├── evaluate_real.py              ← 20-grasp real-world eval
│   └── arm_drivers/                  ← Arm-specific adapters
│       ├── base.py                   ← Abstract RobotArm interface
│       ├── so100_driver.py           ← SO-ARM101 / LeRobot adapter
│       └── mock_driver.py            ← Mock arm for offline testing
│
├── vision/
│   ├── object_detector.py            ← YOLOv8 detection wrapper
│   └── calibration.npz              ← Saved camera calibration
│
├── analysis/
│   ├── plot_results.py               ← Generate comparison charts
│   ├── sim_vs_real.ipynb             ← Notebook: sim ↔ real comparison
│   └── ablation_study.ipynb          ← Notebook: randomisation range sweep
│
├── models/                           ← Saved policy checkpoints
├── logs/                             ← TensorBoard + W&B logs
├── urdf/                             ← Robot URDF model files
├── docker/
│   └── Dockerfile                    ← Cloud training image
├── tests/                            ← Unit & smoke tests
│
├── configs/                          ← Top-level config overrides
├── requirements.txt
├── README.md
├── PLAN.md                           ← This file
└── report/
    └── final_report.pdf
```

---

## Compute Split

| Task | Where | Why |
|---|---|---|
| RL training (millions of steps) | ☁️ Cloud GPU | 512 parallel envs need VRAM |
| Physics simulation | ☁️ Cloud GPU (IsaacGym) or CPU (PyBullet) | Speed |
| Policy inference at runtime | 💻 Laptop (RTX 3060) | Real-time, < 5 ms/step |
| Servo motor commands | 💻 Laptop → USB → Arm | Direct hardware link |
| Camera feed + YOLO | 💻 Laptop | Low latency |
| Experiment tracking | 💻 Laptop + W&B (cloud dashboard) | Logging |

---

## Phased Plan & Progress

<!-- UPDATE THIS FILE AFTER EACH STEP IS COMPLETED -->

---

### Phase 0 — Project Setup ✅ COMPLETE

- [x] Create project folder structure
- [x] Initialise Git repository
- [x] Write initial PLAN.md
- [x] Create `requirements.txt` with all dependencies
- [x] Set up base `Dockerfile` for cloud training
- [x] Install core Python packages (gymnasium, sb3, pybullet, wandb, ultralytics)

---

### Phase 1 — Simulation Environment ✅ COMPLETE

- [x] Generate 6-DOF arm URDF with parallel gripper (inline template in `so100_env.py`)
- [x] Create PyBullet physics environment (`sim/envs/so100_env.py`)
  - [x] 21-dim observation space (joints + EE + object + relative)
  - [x] 6-dim continuous action space (mapped to joint targets)
  - [x] Scene: table, floor, graspable box object
  - [x] Reach task reward
  - [x] Grasp task reward (proximity + lift bonus)
  - [x] Episode termination logic (success at lift height / object fall-off)
- [x] Build domain randomisation module (`sim/envs/domain_rand.py`)
  - [x] `RandomizationConfig` dataclass with all parameter ranges
  - [x] `RandomizationState` per-episode sampled values
  - [x] `DomainRandomizer` — sampling, observation noise, action noise
  - [x] Five preset levels: `none / low / medium / high / extreme`
- [x] Build reward module (`sim/envs/reward.py`)
  - [x] `compute_reach_reward()` with potential-based shaping
  - [x] `compute_grasp_reward()` with gripper-close incentive
  - [x] `AdaptiveReward` auto-scaling based on rolling success rate
- [x] Smoke test — env runs for 4 configs without crash

---

### Phase 2 — Training Pipeline

- [ ] Create YAML config files
  - [ ] `sim/configs/baseline.yaml` — no randomisation, SAC, 5 M steps
  - [ ] `sim/configs/domain_rand.yaml` — medium randomisation, SAC, 5 M steps
  - [ ] `sim/configs/ablation.yaml` — parameterised randomisation levels
- [ ] **Reach task — baseline**
  - [ ] Train SAC for 500 K steps (PyBullet, local or cloud)
  - [ ] Verify agent reaches target > 90 % of episodes
  - [ ] Log to W&B, confirm learning curve rises
- [ ] **Grasp task — baseline (no randomisation)**
  - [ ] Train SAC for 5 M steps on cloud GPU
  - [ ] Use 4–16 parallel SubprocVecEnv workers
  - [ ] Checkpoint every 500 K steps
  - [ ] Evaluate in sim: 100 episodes, record success rate
- [ ] **Grasp task — domain randomised (medium)**
  - [ ] Same architecture & hyperparameters as baseline
  - [ ] Enable `randomization=medium`
  - [ ] Train for 5 M steps, log to W&B
  - [ ] Evaluate in sim: 100 episodes
- [ ] **Compare learning curves** (baseline vs domain-rand) in W&B
- [ ] Save best policies to `models/`

---

### Phase 3 — Ablation Study (Sim)

- [ ] Train additional policies with `randomization=low`, `high`, `extreme`
  - [ ] Same hyperparameters, 5 M steps each
- [ ] Evaluate all 5 variants in sim (100 episodes each)
- [ ] Record per-variant:
  - [ ] Success rate
  - [ ] Mean reward
  - [ ] Mean episode length
- [ ] Plot: **Randomisation Level vs Sim Success Rate**
- [ ] Identify "sweet spot" randomisation level

---

### Phase 4 — Real Robot Hardware Setup

- [ ] **Acquire 6-DOF arm** (SO-ARM101 recommended)
  - [ ] Source parts / order kit
  - [ ] 3D-print structural components (if DIY)
  - [ ] Assemble arm, flash servo firmware
- [ ] **Software integration**
  - [ ] Install arm SDK / LeRobot library
  - [ ] Implement `real/arm_drivers/so100_driver.py` adapter
    - [ ] `get_joint_positions()` → 6-element array
    - [ ] `get_joint_velocities()` → 6-element array
    - [ ] `get_end_effector_position()` → 3-element array (FK)
    - [ ] `set_joint_targets(action)` → send to servos
    - [ ] `gripper_force()` → force estimate
    - [ ] `go_home()`, `enable_torque()`, `disable_torque()`
  - [ ] Verify: command each joint from Python, confirm range matches sim
- [ ] **Workspace setup**
  - [ ] Mount arm on table with clamp
  - [ ] Mark workspace boundary with tape
  - [ ] Place foam blocks at known positions
  - [ ] Measure real link lengths → validate URDF accuracy
- [ ] **Camera calibration**
  - [ ] Mount webcam overhead with clamp
  - [ ] Run `real/calibrate_camera.py --mode calibrate`
  - [ ] Click 4+ table reference points, enter measured world coords
  - [ ] Verify with `--mode test` — hover cursor, confirm world XY readout
  - [ ] Save to `vision/calibration.npz`
- [ ] **Object detection**
  - [ ] Fine-tune or validate YOLOv8n on foam block colours
  - [ ] Confirm detection is stable under workspace lighting
  - [ ] Measure detection latency (target < 30 ms)
- [ ] **End-to-end dry run**
  - [ ] Load a sim-trained policy
  - [ ] Run `real/inference.py --model models/baseline/final_model.zip --attempts 1`
  - [ ] Confirm observation → policy → action → servo loop works
  - [ ] Tune control frequency (target 10–30 Hz)

---

### Phase 5 — Real-World Evaluation

- [ ] **Experiment 1 — Baseline**
  - [ ] Load `policy_baseline.zip`
  - [ ] Run 20 grasp attempts (`real/evaluate_real.py --attempts 20`)
  - [ ] Record every attempt on video (phone / screen capture)
  - [ ] Log: attempt #, success/fail, steps to grasp, notes
- [ ] **Experiment 2 — Domain Randomised**
  - [ ] Load `policy_domain_rand.zip`
  - [ ] Same 20-attempt protocol, same object placements
  - [ ] Record on video
- [ ] **Experiment 3 — Ablation on real hardware**
  - [ ] Load `policy_low.zip`, `policy_high.zip` (and optionally `extreme`)
  - [ ] 20 attempts each, same protocol
  - [ ] Record on video
- [ ] **Consistency check**
  - [ ] Use identical object positions for all policies (mark grid on mat)
  - [ ] Control for lighting — run under same conditions
  - [ ] Note ambient temperature (affects servo behaviour)

---

### Phase 6 — Analysis & Reporting

- [ ] **Quantitative analysis**
  - [ ] Compute per-policy: success rate, mean steps, std dev
  - [ ] Compute **sim-to-real transfer gap** = sim_success − real_success
  - [ ] Build results table (like Expected Outcome above)
- [ ] **Plots** (in `analysis/plot_results.py`)
  - [ ] Bar chart: Sim vs Real success rate per policy
  - [ ] Line chart: Randomisation level vs transfer gap
  - [ ] Learning curves comparison (from W&B export)
  - [ ] Ablation scatter: randomisation range vs real success
- [ ] **Statistical tests**
  - [ ] Binomial confidence intervals on success rates
  - [ ] Chi-squared or Fisher exact test: baseline vs domain-rand
- [ ] **Write final report** (`report/final_report.pdf`)
  - [ ] Introduction + related work (sim-to-real, domain rand)
  - [ ] Method (env design, reward, randomisation, hardware)
  - [ ] Results (tables + figures)
  - [ ] Discussion (what worked, failure modes, limitations)
  - [ ] Conclusion + future work
- [ ] **Create demo video**
  - [ ] Side-by-side: sim training → real deployment
  - [ ] Show baseline failures vs domain-rand successes
  - [ ] Narration / captions explaining what's happening
- [ ] **Clean up repository**
  - [ ] Remove debug code, add docstrings
  - [ ] Update README.md with final instructions
  - [ ] Tag release in Git

---

## Week-by-Week Timeline

| Week | Focus | Deliverables |
|---|---|---|
| **1** ✅ | Project setup, folder structure, dependencies | PLAN.md, requirements.txt, Dockerfile |
| **2** ✅ | Simulation environment in PyBullet | `so100_env.py`, domain_rand, reward.py, smoke tests passing |
| **3** | YAML configs, reach task baseline training | `baseline.yaml`, reach policy > 90 % in sim |
| **4** | Grasp task baseline training (cloud) | Baseline grasp policy, W&B dashboard live |
| **5** | Domain-randomised grasp training (cloud) | DR policy trained, learning curves compared |
| **6** | Ablation: train low / high / extreme variants | All 5 policies saved in `models/` |
| **7** | Sim evaluation of all policies (100 eps each) | Evaluation JSON results, sim comparison chart |
| **8** | Order / build real arm, start assembly | Arm assembled, servos responding |
| **9** | Arm SDK integration, Python joint control verified | Driver adapter passing all checks |
| **10** | Camera calibration, YOLO validation, workspace setup | `calibration.npz`, detection stable |
| **11** | End-to-end dry run, tune control frequency | Full loop works on real arm |
| **12** | Real eval — Experiment 1 (baseline) | 20 grasps + video, logged results |
| **13** | Real eval — Experiments 2 & 3 (DR + ablation) | 60+ grasps + video, logged results |
| **14** | Data analysis, plots, statistical tests | Figures + results table |
| **15** | Write report, create demo video | Draft report, demo video |
| **16** | Polish & submit | Final report, clean repo, presentation |

---

## Quick-Start Commands

### Local development (CPU, PyBullet)
```bash
# Install dependencies
pip install -r requirements.txt

# Smoke-test the environment
python -c "from sim.envs.so100_env import SO100Env; e=SO100Env('reach'); e.reset(); print('OK')"

# Quick reach training (local, few steps)
cd sim && python train.py --task reach --steps 100000 --num_envs 2 --no-wandb
```

### Cloud training (GPU)
```bash
# Build Docker image
docker build -t sim-to-real -f docker/Dockerfile .

# Baseline grasp training
docker run --gpus all sim-to-real \
  python sim/train.py --task grasp --randomization none --steps 5000000 --num_envs 16

# Domain-randomised grasp training
docker run --gpus all sim-to-real \
  python sim/train.py --task grasp --randomization medium --steps 5000000 --num_envs 16
```

### Real robot inference
```bash
# Calibrate camera
python real/calibrate_camera.py --mode calibrate --checkpoint vision/calibration.npz

# Run trained policy on real arm
python real/inference.py --model models/grasp_medium/final_model.zip --attempts 20

# Full evaluation with logging
python real/evaluate_real.py --model models/grasp_medium/final_model.zip --attempts 20 --output results/
```

---

## Key Design Decisions

1. **Arm-agnostic architecture** — the RL training knows nothing about the
   specific arm brand.  It sees joints ∈ ℝ⁶ and a URDF.  The real-world
   driver is a swappable adapter behind an abstract `RobotArm` interface.

2. **PyBullet first, IsaacGym second** — PyBullet lets us iterate locally
   without a GPU.  When it's time for million-step training, we switch to
   IsaacGym on cloud for 10–100× speed via massively-parallel envs.

3. **SAC as default algorithm** — SAC is sample-efficient and handles
   continuous action spaces well.  PPO is available as a fallback (better
   on IsaacGym's GPU-vectorised envs).

4. **Hand-crafted reward over learned reward** — keeps the project
   tractable for a semester.  The reward is transparent and easy to debug.

5. **Domain randomisation over domain adaptation** — DR is simpler to
   implement, doesn't require real-world data, and is well-supported by
   the literature for closing the sim-to-real gap.

6. **SO-ARM101 as recommended hardware** — it has the best RL ecosystem
   support of any affordable 6-DOF arm.  But the codebase doesn't hard-code
   the arm — any 6-DOF arm with a Python serial SDK can be plugged in.

---

## Key Files

| File | Purpose |
|---|---|
| `sim/envs/so100_env.py` | PyBullet 6-DOF arm environment (obs, action, step, reward) |
| `sim/envs/domain_rand.py` | Randomisation configs, sampling, noise injection |
| `sim/envs/reward.py` | Reward functions (reach, grasp) + adaptive reward |
| `sim/train.py` | Training script (SAC/PPO, parallel envs, W&B, checkpoints) |
| `sim/evaluate_sim.py` | Evaluate policy over N sim episodes |
| `real/inference.py` | Load policy → run on real arm with camera |
| `real/calibrate_camera.py` | Interactive camera ↔ world calibration tool |
| `real/evaluate_real.py` | Structured 20-attempt real-world evaluation |
| `analysis/plot_results.py` | Generate comparison charts for report |

---

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Arm delivery delay | Blocks Phase 4–5 | All sim work is independent; order early |
| Sim-to-real gap too large even with DR | Weakens results | Add action smoothing, observation filtering; tune URDF to match real arm precisely |
| Servo overheating during long eval sessions | Hardware damage | Add cool-down pauses between attempts; monitor servo temps |
| YOLO fails under workspace lighting | Blocks inference | Use colour-threshold fallback; control lighting with desk lamp |
| Cloud GPU costs exceed budget | Budget overrun | Use spot instances; limit to 50 hrs; train in smaller batches |
| PyBullet contact physics too inaccurate | Poor transfer | Migrate to MuJoCo for better friction/contact modelling |