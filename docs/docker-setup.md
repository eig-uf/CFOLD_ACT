# Docker development

The default image provides Python **3.11.14**, uv **0.9.11**, Debian Bookworm,
and the existing `uv.lock` dependencies with the `test` extra. It retains the
explicit CPU PyTorch index: torch 2.7.1, torchvision 0.22.1, LeRobot 0.4.4 with
Feetech support, and TorchCodec 0.5. No host Python installation is used.

FFmpeg provides the video encoder/decoder and shared libraries required by
TorchCodec. Bookworm's FFmpeg 5 is within TorchCodec's supported FFmpeg range;
see the [TorchCodec installation guide](https://github.com/pytorch/torchcodec/tree/v0.5.0#installing-torchcodec).
`libgomp1` provides the numerical runtime; `v4l-utils` supports camera discovery.
GCC, libc development headers and Linux input headers build the locked `evdev`
C extension pulled in by LeRobot's `pynput` dependency.
LeRobot supplies headless OpenCV, Pillow, NumPy, PySerial and the Feetech SDK
through uv. No extra pip-installed packages or host system libraries are used.

## Host prerequisites

Use an x86-64 laptop, Git, VS Code, and a running Docker Engine with Linux
containers. The first build needs internet access to the container registries,
Debian repositories, PyPI, and download.pytorch.org, plus several GB of disk
space. No NVIDIA GPU, robot, or camera is required for development and tests.

### Windows with WSL2 and Docker Desktop

In an administrator PowerShell terminal:

```powershell
wsl --install -d Ubuntu
wsl --update
wsl --set-default-version 2
winget install -e --id Docker.DockerDesktop
winget install -e --id Microsoft.VisualStudioCode
```

Restart when prompted and complete Ubuntu's initial user setup. Start Docker
Desktop, select the WSL2 engine in Settings, and enable Ubuntu under
**Resources > WSL Integration**. Use Linux containers. See Docker's
[Windows installation instructions](https://docs.docker.com/desktop/setup/install/windows-install/).
Do not install a second Docker daemon inside Ubuntu when using Desktop's engine.

Install these extensions in Windows VS Code:

```powershell
code --install-extension ms-vscode-remote.remote-wsl
code --install-extension ms-vscode-remote.remote-containers
```

Run the clone/open commands below in the Ubuntu terminal. Keep the repository
under `~/projects`, rather than `/mnt/c`, for Linux permissions and filesystem
performance. Install Git there with `sudo apt-get update && sudo apt-get install -y git`
if needed. `code .` opens the WSL workspace in Windows VS Code.

### Arch Linux

Install the [Arch Docker package](https://archlinux.org/packages/extra/x86_64/docker/)
and Buildx (Compose is not required):

```bash
sudo pacman -Syu docker docker-buildx git
sudo systemctl enable --now docker.service
sudo usermod -aG docker "$USER"
```

Log out and back in after adding the group. Docker group membership permits
administrative control of the host. Install VS Code using your preferred package
source, then install Dev Containers:

```bash
code --install-extension ms-vscode-remote.remote-containers
docker version
docker run --rm hello-world
```

For other Linux distributions, follow the distribution-specific
[Docker Engine installation guide](https://docs.docker.com/engine/install/).

## Clone and open in VS Code

From Linux or your WSL Ubuntu terminal:

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/eig-uf/CFOLD_ACT.git
cd CFOLD_ACT
code .
```

Run **Dev Containers: Reopen in Container** from the command palette. VS Code
builds the repository Dockerfile locally and synchronizes with
`uv sync --locked --extra test` when creating and starting the container.
The first build downloads the ML dependencies and can take several minutes.
A stale lockfile fails synchronization rather than silently resolving new versions.

The repository is bind-mounted at `/workspaces/CFOLD_ACT`: edits persist on the
host immediately. The editable project installation uses this same path. The
virtual environment is `/home/vscode/.venv`, outside the mount, so an existing
Windows/Linux host `.venv` cannot replace container dependencies. The environment
is built into the image; no persistent dependency volume can conceal an update.
PATH and VS Code's Python interpreter point to this environment automatically.
Python, Pylance, and the Python debugger extensions are installed in the container.
Pytest discovery reads `testpaths` and `pythonpath` from `pyproject.toml`.

Processes run as `vscode`, not root. On native Linux, Dev Containers adjusts the
user's UID/GID to match the host; its home directory (including the environment)
is adjusted too. See [VS Code's non-root user documentation](https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user).
For an existing VS Code workspace with a saved interpreter, use **Python: Select
Interpreter** and choose `/home/vscode/.venv/bin/python` once.

## Run and test

In a container terminal, from the repository root:

```bash
python --version
uv --version
uv sync --locked --extra test
uv run pytest
uv run python -m src.deploy --help
uv run python -m src.cli train --stage 1 --print-config
uv run python -c 'import torch, torchvision, torchcodec, lerobot, cv2, numpy, yaml; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())'
```

The checked-in tests use fake motor/camera interfaces and small randomly
initialized CPU ACT models, so the normal suite needs no devices or downloaded
checkpoints. The shared unit-test fixture replaces sysfs camera identification
with path identification: host webcams can otherwise make the example video
nodes appear to be duplicate interfaces of one camera, even without device
passthrough. Application hardware validation is unchanged.
`test_trained_checkpoint_inference` already skips unless
`CFOLD_CHECKPOINT` points to a compatible trained checkpoint; this is an optional
model test, not a physical hardware test. Supply checkpoints separately; they
are excluded from image builds. No tests have been removed or newly skipped.

Actual recording, teleoperation, deployment and `scripts/test_cameras.py` need
devices and calibration/checkpoints as described in [OPERATIONS.md](OPERATIONS.md).
The camera test should use `--no-display` in this headless container. Tk and
desktop forwarding are intentionally omitted; use the existing local workflow
for GUI previews, or configure them separately. OpenCV's headless package must
not be replaced with a second cv2 package.

Training defaults to CUDA in the existing scripts. For CPU development, explicitly
choose CPU, for example `DEVICE=cpu STEPS=10 bash scripts/train_fold1.sh`.
Training also needs a dataset and may download pretrained backbone weights;
these data and downloads are separate from installing dependencies.

## Develop on Linux without VS Code

Use any host editor (for example Vim, Emacs, or your preferred IDE) and run Python
commands inside Docker. Only Git and Docker are required on the host; VS Code,
the Dev Containers extension, host Python, and host uv are optional.

Clone the repository and enter its root directory, or use your existing checkout:

```bash
git clone https://github.com/eig-uf/CFOLD_ACT.git
cd CFOLD_ACT
```

Build the same development image, matching its non-root user to your host UID/GID
so container-created files remain editable on Linux:

```bash
docker build \
  --build-arg USER_UID="$(id -u)" \
  --build-arg USER_GID="$(id -g)" \
  -t cfold-act:dev .
docker run --rm cfold-act:dev python --version
docker run --rm cfold-act:dev uv --version
```

Start a development shell from the repository root:

```bash
docker run --rm -it \
  --mount "type=bind,source=$PWD,target=/workspaces/CFOLD_ACT" \
  cfold-act:dev bash -c \
  'uv sync --locked --extra test && exec bash'
```

Edit the repository with your host editor while this shell is open. The bind
mount makes source changes visible immediately in both directions. Inside the
container shell, run:

```bash
uv run pytest
uv run python -m src.deploy --help
uv run python -m src.cli train --stage 1 --print-config
```

For a single command without an interactive shell, mount the current checkout
so the test run includes your latest edits:

```bash
docker run --rm \
  --mount "type=bind,source=$PWD,target=/workspaces/CFOLD_ACT" \
  cfold-act:dev bash -c 'uv sync --locked --extra test && uv run pytest'
```

Use `exit` to leave the shell. `--rm` removes that container, while source edits,
outputs, and dependency-file changes in the mounted repository persist on the
host. The built image remains available; rerun the shell command next time.
Changes elsewhere in the container, including dependencies added to its virtual
environment, disappear when it is removed. After `uv add` or dependency updates,
commit the manifest/lockfile changes and repeat the UID/GID-aware build command
above to bake the updated environment into the image. Source-only edits need
no rebuild. Never bind-mount a host virtual environment over the container
environment. Use Git from your host terminal as usual.

If your `docker` command uses rootless Podman, add `--userns=keep-id` to each
bind-mounted `docker run` command so the non-root user can write to the host
checkout. This option is specific to Podman; omit it with Docker Engine/Desktop.
These command-line instructions also work from a WSL Linux terminal.

## Dependencies, rebuilds and a second laptop

Inside either the Dev Container or the command-line development shell:

```bash
uv add PACKAGE_NAME
uv add --optional test TEST_PACKAGE_NAME
uv lock --upgrade-package PACKAGE_NAME
uv sync --locked --extra test
uv run pytest
git diff -- pyproject.toml uv.lock
```

Run only the add/upgrade command needed for your change. Review and commit both
`pyproject.toml` and `uv.lock` when either changes. Avoid `pip install`, `uv pip
install`, and unrequested full dependency upgrades: project dependencies belong
in the manifest and lockfile. The existing CPU PyTorch sources stay authoritative.
`uv run --locked --extra test pytest` is useful in automation to reject lock drift.

After dependency or Dockerfile changes, select **Dev Containers: Rebuild
Container**. Source-only changes need no rebuild. A restart synchronizes locked
Python dependencies, but changing the base image or apt packages requires a
rebuild. For command-line use, repeat the UID/GID-aware build command in the
section above and create a new container; existing containers do not update
themselves.

Commit work and share it through Git using your normal workflow. On laptop two,
install the host prerequisites, clone the same branch/commit, and reopen in the
container, or build and start the command-line shell above. For an existing
checkout:

```bash
git pull --ff-only
```

Then rebuild the image and start a new command-line shell. If using VS Code,
run `code .` and select **Dev Containers: Rebuild Container** instead.
Do not copy `.venv`, images,
caches, or host Python packages between machines. Datasets, trained checkpoints,
calibration files and credentials must be transferred/configured separately;
keep secrets out of Git and image layers.

## Optional NVIDIA GPU environment

CPU inference is supported; CUDA can accelerate ACT training and inference.
Passing `--gpus all` alone cannot enable CUDA in the locked CPU torch wheels.
The default image and lockfile deliberately remain CPU-only.

On native Linux, install a compatible NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host, then configure Docker:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
nvidia-smi
```

On Windows, use a supported NVIDIA Windows driver, updated WSL2 and Docker
Desktop's WSL2 backend. Follow [Docker Desktop GPU prerequisites](https://docs.docker.com/desktop/features/gpu/);
do not install a Linux display driver inside the container/WSL distribution.

For an optional CUDA variant, keep a **separate Git worktree and branch with its
own manifest and lockfile**. This changes the torch source intentionally while
keeping torch/torchvision versions at 2.7.1/0.22.1. PyTorch publishes a matching
[CUDA 12.6 wheel pair](https://pytorch.org/get-started/previous-versions/#v271).
Build the CPU image first, then:

```bash
git worktree add -b dev/cuda ../CFOLD_ACT-cuda
cd ../CFOLD_ACT-cuda
# Resolve using the pinned container tools, not a host uv installation.
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  --mount "type=bind,source=$PWD,target=/workspaces/CFOLD_ACT" \
  cfold-act:dev python -c 'from pathlib import Path; p = Path("pyproject.toml"); s = p.read_text(); p.write_text(s.replace("pytorch-cpu", "pytorch-cuda").replace("https://download.pytorch.org/whl/cpu", "https://download.pytorch.org/whl/cu126"))'
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  --mount "type=bind,source=$PWD,target=/workspaces/CFOLD_ACT" \
  cfold-act:dev uv lock --python /usr/local/bin/python3.11
git diff -- pyproject.toml uv.lock
docker build -t cfold-act:cuda .
docker run --rm --gpus all cfold-act:cuda uv run --locked python -c \
  'import torch, torchvision, torchcodec; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.ones(1, device="cuda"))'
docker run --rm --gpus all cfold-act:cuda uv run --locked --extra test pytest
```

Review, test, and commit the CUDA variant on its separate branch before sharing
it. In that worktree's Dev Container configuration, add
`"runArgs": ["--gpus", "all"]` at the top level and rename the container to
identify it as CUDA, then rebuild. `uv sync --locked` now preserves that variant
because its own lockfile records CUDA sources. Do not merge its source changes
into the default CPU branch. This optional recipe requires GPU validation on
the target host; it is not a prevalidated CUDA lockfile.

Run training with `DEVICE=cuda`, or deployment with `--device cuda`. Keep
TorchCodec CPU video decoding unless GPU decoding is separately needed and
validated: NVDEC has additional GPU, FFmpeg and TorchCodec build requirements.
No CUDA compiler/toolkit is installed in the development image; prebuilt torch
wheels provide their userspace CUDA runtime. Custom CUDA extensions need a
separate toolchain. Driver compatibility and GPU architecture remain host-specific.

## Optional robot and camera passthrough

On native Linux, map only the actual device nodes needed. For example, two
followers and three cameras (substitute the nodes discovered on your machine):

```bash
docker run --rm -it \
  --device /dev/ttyACM0 --device /dev/ttyACM1 \
  --device /dev/video0 --device /dev/video2 --device /dev/video4 \
  --group-add "$(stat -c '%g' /dev/ttyACM0)" \
  --group-add "$(stat -c '%g' /dev/video0)" \
  --mount "type=bind,source=$PWD,target=/workspaces/CFOLD_ACT" \
  cfold-act:dev bash
```

Build the image with your UID/GID as shown above. Add every distinct device
group GID with `--group-add`.
Teleoperation also needs the two leader serial devices (`ttyACM2/3` in the
example configuration). `scripts/find_cameras.sh` uses V4L2; UVC camera support
and USB serial drivers come from the **host kernel**, not apt packages inside
the container. These interfaces use serial/V4L2 nodes, so mounting the entire
USB bus is unnecessary. A RealSense or other camera backend would need its own
optional dependency and device configuration; none is configured by this project.

For VS Code, add equivalent `--device=...` and `--group-add=NUMERIC_GID` entries
to a local `runArgs` array and rebuild. Device names and group IDs vary by host;
keep those edits local. The default configuration has no devices, host Docker
socket mount, or `--privileged` access.

Persistent aliases are recommended by [OPERATIONS.md](OPERATIONS.md), but a
`--device` mapping alone does not create `/dev/serial/by-id` or `/dev/v4l/by-path`
symlinks in the container. Either use the mapped canonical device paths in a
local robot YAML, or mount the alias directories read-only at the same paths
and map their resolved nodes too. Pass a local config with `--robot-config` or
`ROBOT_CONFIG`. Make calibration files available in the mounted workspace or a
dedicated bind mount. Devices reconnected after container creation may need a
container restart; numeric device assignments can change.

Windows USB passthrough requires an extra host setup, not merely `--device`.
Use [Microsoft's USB/IP instructions](https://learn.microsoft.com/en-us/windows/wsl/connect-usb)
to install usbipd-win, bind a device from elevated PowerShell, and attach it:

```powershell
winget install --interactive --exact dorssel.usbipd-win
usbipd list
usbipd bind --busid 4-4
usbipd attach --wsl --busid 4-4
```

Replace `4-4` with the listed bus ID. Confirm the serial/V4L2 node exists in the
Linux environment used by the **Docker daemon**, not just your Ubuntu shell,
before applying device mappings. WSL camera drivers, USB/IP camera bandwidth,
and Docker Desktop device visibility vary; attaching to WSL does not guarantee
Desktop can expose the device. Use native Linux for robot operation if the
required devices are unavailable. CPU development remains independent of USB.

## Reproducibility limits and validation

Python and uv images are pinned by version and multiarchitecture digest, and uv
checks the committed lockfile, including artifact hashes. Project dependencies
are installed before source files for layer caching. Rebuilding restores the
environment; updating image digests is a deliberate reviewed change.

Builds are not bit-for-bit reproducible: apt repositories receive updates,
Python's isolated build tools can change within manifest constraints, build
timestamps/caches differ, and compiled wheels differ by CPU architecture.
The supported baseline is Linux x86-64 on both host platforms. Although the
base images are multiarchitecture, the locked robotics/ML wheels may not cover
ARM; ARM builds are unvalidated. Emulating `linux/amd64` is possible with Docker
emulation but slower, particularly for ML. GPU drivers and numerical kernels
also affect behavior and determinism. VS Code extensions are not version-locked.

To check a new laptop, build the image and run the version/import/test commands
above, then create a file in the mounted container workspace and confirm it
appears on the host. `id` should show a non-root user. Run
`git status --short` and `git diff --check` to inspect the intended changes;
virtual environments, caches, datasets, outputs and secrets should stay untracked.
The Dev Container JSON can be checked with
`python -m json.tool .devcontainer/devcontainer.json` inside the container.
VS Code's actual reopen/rebuild tests extension and host integration.

### Validation performed for this setup

On Linux x86-64 using Podman's Docker-compatible CLI, the image built, Python
3.11.14 and uv 0.9.11 were verified, and a subsequent locked sync succeeded.
Imports passed for torch/torchvision (CPU builds), TorchCodec, LeRobot, OpenCV,
NumPy, PyYAML, PySerial, the Feetech SDK, and project modules. The deployment
help and training configuration commands also ran.

The official Dev Containers CLI 0.80.0 successfully read the configuration,
built and started the container, and ran both lifecycle synchronization commands.
The non-root user could write to the environment and bind-mounted workspace;
a container-created file was confirmed on the host and then removed.
`uv run pytest` against the mounted final source reported **47 passed, 1 skipped**
(the existing optional trained-checkpoint test).

The first build identified missing evdev build headers, which are now installed.
The initial tests exposed the host sysfs camera-topology dependency described
above; the shared fixture now isolates it. No tests were removed. A plain
Podman bind-mount command also needed `--userns=keep-id` for write permissions;
the Dev Containers CLI detected Podman and supplied this automatically. The
command-line examples target Docker Engine/Desktop.

Actual VS Code GUI reopening/extension installation, Windows/WSL2, ARM, CUDA,
and physical robot/camera operation were not tested in this environment.
