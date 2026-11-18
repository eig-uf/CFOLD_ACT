# CFOLD ACT

A hierarchical imitation learning system for bimanual cloth folding using independently trained Action Chunking Transformer (ACT) policies.

## Development environment

For a shared Python 3.11 CPU environment on Windows/WSL2 and Linux, open this
repository in VS Code and select **Dev Containers: Reopen in Container**.
Docker builds the locked dependencies locally, including the test extra.
Run `uv run pytest` in the container terminal.

See [Docker and Dev Container setup](docs/docker-setup.md) for host installation,
command-line usage, dependency updates, switching laptops, and optional NVIDIA
GPU and robot device access. See [operations](docs/OPERATIONS.md) for application,
training, recording, and hardware configuration commands.

## System Architecture

```text
Observations
├── RGB: overhead, left wrist, right wrist
└── Proprioception: joint positions
                  │
                  ▼
         Task State Machine
          ┌───────┴───────┐
          ▼               ▼
       ACT π₁          ACT π₂
     First fold      Second fold
          └───────┬───────┘
                  ▼
          Action Chunk (K × 12)
                  │
          Action Execution
          ┌───────┴───────┐
          ▼               ▼
       Left Arm        Right Arm
```

## Policy Architecture

Each ACT policy jointly controls both manipulators, mapping visual and proprioceptive observations to sequences of coordinated motor position targets.

### Observations

The observation at timestep $t$ consists of three RGB images and 12 measured motor positions:

$$
s_t = (I_t^{overhead}, I_t^{left}, I_t^{right}, q_t)
$$

$$
q_t \in \mathbb{R}^{12}
$$

### Actions

Each policy predicts a chunk of $K$ future actions, where each action contains 12 absolute motor position targets.

$$
\hat{A}_t = \pi_\theta(s_t) \in \mathbb{R}^{K \times 12}
$$

$$
\hat{A}_t = [\hat{a}_t,\hat{a}_{t+1},\ldots,\hat{a}_{t+K-1}]
$$

## Task Decomposition

The folding task is decomposed into two policies with identical architectures but independent weights.

| Policy | Initial State | Objective |
|---|---|---|
| $\pi_{\theta_1}$ | Unfolded cloth | Fold into rectangle |
| $\pi_{\theta_2}$ | Half-folded cloth | Fold into square |

Both policies control the complete bimanual system:

$$
\pi_{\theta_i}: \mathcal{S} \rightarrow \mathcal{A}^{K}
$$
