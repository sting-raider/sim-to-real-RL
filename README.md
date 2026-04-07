# SO-100 Sim-to-Real Arm Project

A robotic arm that learns to grasp and manipulate objects inside physics simulation, then gets deployed onto a real SO-100 arm with zero additional training.

## Quick Start

### Development (Local)
```bash
pip install -r requirements.txt
python sim/train.py --task reach --steps 500000 --randomization none
```

### Cloud Training
```bash
docker build -t sim-to-real-arm -f docker/Dockerfile .
docker run --gpus all sim-to-real-arm python sim/train.py --task grasp --steps 5000000
```

## Project Structure
```
sim-to-real-arm/
├── sim/                    # Simulation environment and training
│   ├── envs/               # Gym environments
│   │   ├── so100_env.py    # SO-100 robot arm environment
│   │   ├── domain_rand.py  # Domain randomization module
│   │   └── reward.py       # Reward functions
│   ├── configs/            # Training configurations
│   ├── train.py            # Main training script
│   └── evaluate_sim.py     # Simulation evaluation
├── real/                   # Real robot deployment
│   ├── inference.py        # Real robot inference pipeline
│   ├── calibrate_camera.py # Camera calibration tool
│   └── evaluate_real.py    # Real-world evaluation
├── vision/                 # Vision components
├── analysis/               # Results visualization
│   └── plot_results.py     # Plotting scripts
├── models/                 # Trained policy checkpoints
├── docker/                 # Docker configuration
├── requirements.txt
└── PLAN.md                 # Project plan and progress
