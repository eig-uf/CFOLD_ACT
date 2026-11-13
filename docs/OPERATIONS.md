# CFOLD operations

README.md is unchanged. This implementation targets **LeRobot 0.4.4**, Python
3.11–3.12, PyTorch 2.7.1, torchvision 0.22.1 and torchcodec 0.5. `uv.lock`
records the tested CPU environment. Hardware, trained folding performance and
Framework laptop throughput have not been validated.

## Install and inspect

```bash
uv sync --python 3.12 --extra test
source .venv/bin/activate
python -m pytest -q
python -m src.deploy --help
```

LeRobot supplies the motor, camera, dataset, ACT and processor implementations.
The integration was checked against the [v0.4.4 source](https://github.com/huggingface/lerobot/tree/v0.4.4/src/lerobot),
including `robots/bi_so_follower`, `teleoperators/bi_so_leader`,
`policies/act`, `policies/factory.py`, `policies/utils.py`,
`datasets/utils.py`, and the record/train/teleoperate scripts. Automated tests
decode generated configurations using the installed native CLI dataclasses.

Training and video decoding need a compatible system FFmpeg runtime; check it
with `python -c 'import torchcodec'`. On Arch, install `v4l-utils` for discovery
and `tk` for the optional camera preview. The preview uses stdlib Tk and
LeRobot's Pillow dependency because LeRobot installs headless OpenCV.

## Configure and validate hardware

Edit `configs/robot.yaml` before any motor operation:

- Replace all four serial ports with persistent `/dev/serial/by-id/` paths.
- Assign the three physical cameras explicitly. Prefer `/dev/v4l/by-id/`, or
  `/dev/v4l/by-path/` for identical InnoMaker wrists without unique serial IDs.
- Set group IDs, calibration directories and optional `calibration_ids` to
  existing calibration files. IDs omit `.json`. Native defaults derive child
  IDs as `<group_id>_left` and `<group_id>_right`. The optional mapping selects
  existing files independently of those IDs, and updates the native bus's
  calibration reference. No calibration is written. Missing files or motor/file
  mismatches fail before native motor configuration.
- Validate position limits in degrees (gripper uses LeRobot's 0–100 scale),
  maximum per-command displacement, camera age and action age. Supplied limits
  are initial examples, not validated workspace geometry. `max_step` rejects
  unsafe group actions; native `max_relative_target` adds measured-position
  clipping. Targets are checked against measured positions and the previous
  commanded target; the command history is cleared at stage boundaries.
- Confirm RGB orientation, MJPG support, 640×480 and 30 FPS on every camera.
  Rotation is 0, 90, 180 or -90, interpreted by LeRobot as output orientation.
  Camera output sizes must match checkpoint features exactly.

```bash
bash scripts/find_cameras.sh
python scripts/test_cameras.py --duration-s 30
# Without a desktop:
python scripts/test_cameras.py --duration-s 30 --no-display
```

Discovery lists device capabilities, model names, formats, resolutions, rates
and persistent aliases. Multiple video nodes can belong to one physical camera.
The configuration rejects duplicate aliases and, when sysfs is available,
multiple nodes on the same USB camera. Check wrist labels visually; software
cannot determine which physical wrist an identical camera is mounted on.

The camera test opens all streams together with native OpenCV camera objects,
checks negotiated dimensions/FPS, and reports effective delivery, capture rate,
buffer age, buffer read latency, host capture wait and estimated skipped frames.
Capture wait times the native hardware read, including driver waiting/buffering.
Buffer age is a software timestamp
measurement, not sensor exposure-to-photon latency. It exits nonzero for slow
delivery or estimated drops. Repeat after USB topology or stream changes.

The v0.4.4 bimanual interface owns cameras per arm. Logical camera names map
explicitly to recorded/raw keys:

| Logical name | Native observation / dataset image suffix |
|---|---|
| overhead | left_overhead |
| left_wrist | left_left_wrist |
| right_wrist | right_right_wrist |

Full dataset keys begin with `observation.images.`. Recording and inference
use this same native schema. `model` is informational; `arm` determines camera
ownership. Neither field is forwarded into OpenCV's configuration. Changing
ownership changes the schema and requires a compatible dataset/checkpoint.

## Teleoperate and record

Support the arms before connection: native LeRobot follower connection briefly
disables torque while configuring motor registers. Physically align each
leader with its follower before enabling teleoperation. A first target outside
`max_step` aborts the entire group; there is no automatic alignment trajectory.

```bash
bash scripts/teleoperate.sh
NUM_EPISODES=20 EPISODE_TIME_S=18 RESET_TIME_S=15 bash scripts/record_fold1.sh
NUM_EPISODES=20 EPISODE_TIME_S=18 RESET_TIME_S=15 bash scripts/record_fold2.sh
```

Start fold 1 episodes with a flat square and consistent arm poses. Start fold 2
episodes with a rectangle; include realistic intermediate outcomes of policy 1
as demonstrations become available. Twenty episodes above is an example, not a
required training set size. Recording defaults to ten episodes and no upload.

Shared environment options:

| Option | Default / purpose |
|---|---|
| `PYTHON` | `python`; select the installed environment |
| `ROBOT_CONFIG` | `configs/robot.yaml` |
| `DATASET_REPO_ID` | `ufeig/cfold-fold1-v1` or `ufeig/cfold-fold2-v1` |
| `DATASET_ROOT` | Optional local LeRobot dataset root |
| `NUM_EPISODES` | 10; configurable |
| `EPISODE_TIME_S`, `RESET_TIME_S` | 18, 15 |
| `DISPLAY_DATA` | false; native Rerun display |
| `VIDEO`, `VCODEC` | true, h264 |
| `STREAMING_ENCODING`, `ENCODER_THREADS` | false, 2 |
| `PUSH_TO_HUB` | false; use true to upload through existing HF authentication |

`--print-config` prints native JSON without opening hardware. Recording accepts
additional verified native CLI options, e.g. `--resume=true` or
`--dataset.video_encoding_batch_size=5`. Hardware overrides belong in YAML so
the CLI and safety guards use the same configuration. Do not override dataset
feature naming or FPS independently of that shared configuration; these overrides
are rejected by the launcher.

The launcher invokes native LeRobot recording/teleoperation loops with a small
guard that prevents recalibration, validates complete group actions before
either arm is commanded, holds on disconnect and locks follower ports. Native
recording writes standard synchronized observation/action frames. Capture timing
is LeRobot's software sampling; the cameras do not have hardware synchronization.
Use the native recording keyboard controls (right arrow ends episode, left
arrow rerecords, Escape stops). Use a graphical session for those controls;
LeRobot disables keyboard listening in headless environments. Ctrl+C also exits.

## Train independently on HiPerGator

The default install is CPU. In a training environment with a compatible NVIDIA
driver, replace only the torch wheels with the matching CUDA build, for example:

```bash
uv pip install --python .venv/bin/python --no-sources \
  --index-url https://download.pytorch.org/whl/cu126 \
  'torch==2.7.1' 'torchvision==0.22.1'
.venv/bin/python -c 'import torch; print(torch.cuda.is_available())'
```

Run training directly in that environment; `uv sync` restores the project's CPU
wheel selection. Choose a CUDA wheel/driver combination supported by the cluster.
No partition, account or QOS is embedded in the job script.

```bash
DEVICE=cuda BATCH_SIZE=8 STEPS=50000 SAVE_FREQ=5000 bash scripts/train_fold1.sh
DEVICE=cuda BATCH_SIZE=8 STEPS=50000 SAVE_FREQ=5000 bash scripts/train_fold2.sh
# Supply your own resources, for example:
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION --qos=YOUR_QOS \
  --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=08:00:00 \
  scripts/train_slurm.sh 1
```

Submit from the project directory. Repeat with stage `2`. Environment settings
are inherited by Slurm. Set `PYTHON` when using a cluster-provided environment.

Training options are `DATASET_REPO_ID`, `DATASET_ROOT`, `OUTPUT_DIR`,
`POLICY_REPO_ID`, `DEVICE`, `BATCH_SIZE`, `STEPS`, `SAVE_FREQ`, `PUSH_TO_HUB`,
`SEED` and `NUM_WORKERS`. Defaults are stage-specific dataset/policy IDs,
`outputs/foldN`, CUDA, batch 8, 50,000 steps, 5,000-step saves, upload false,
seed 1000 and four workers. Use a new output directory for a new training run.
ACT uses its default architecture, chunk size 100 and execution horizon 30.
Native training saves weights, model configuration and each policy's own
pre/postprocessor configs and statistics. W&B is disabled.

After training (or a Python-observed subprocess failure with saved checkpoints),
the wrapper copies dataset `meta/info.json` into every `pretrained_model` as
`dataset_info.json`. When upload is enabled it adds this metadata to the model
repository after native model/processor uploads. An uncatchable cluster kill
can interrupt that final packaging; before deploying such a checkpoint copy the
matching dataset's `meta/info.json` alongside it manually. Training itself does
not import deployment code or depend on hardware.

Authenticate using the installed Hugging Face login mechanism. No tokens are
stored by this project. For fully offline training, cache/download the dataset
and pretrained torchvision backbone first, then use `DATASET_ROOT`.

## Package and deploy

Copy complete checkpoint directories from HiPerGator, including processor JSON,
processor safetensors and dataset metadata. For locally produced outputs:

```bash
mkdir -p checkpoints/fold1 checkpoints/fold2
cp -a outputs/fold1/checkpoints/last/pretrained_model/. checkpoints/fold1/
cp -a outputs/fold2/checkpoints/last/pretrained_model/. checkpoints/fold2/
python -m src.deploy --robot-config configs/robot.yaml \
  --task-config configs/task.yaml --dry-run
python -m src.deploy --robot-config configs/robot.yaml \
  --task-config configs/task.yaml --device cpu
```

`configs/task.yaml` controls checkpoints, metadata, execution horizons, stage
durations, timeouts, optional action counts, verification and JSONL output.
The initial 12-second durations are placeholders. Local checkpoints are direct
`pretrained_model` contents, not the outer training/checkpoint directory.
`metadata` can point at dataset `meta/info.json`, a dataset directory, or exported
`dataset_info.json`. Set it to null to use the file bundled with the checkpoint.
Motor name ordering comes from that dataset metadata, never dict iteration.
Use the metadata from the actual training dataset: equal dimensions alone do
not establish semantic compatibility with unrelated demonstrations.

Hub IDs work by setting `checkpoint: ufeig/cfold-act-fold1-v1` and a corresponding
second-stage ID. `local_files_only: true` requires a previously cached complete
snapshot; set false to download. Each policy keeps independent processors and
normalization statistics. Local inference skips ImageNet initialization because
the checkpoint already supplies the complete backbone weights; no network is
needed. Requesting CUDA on a CPU-only install fails clearly.

Dry-run defaults to synthetic uint8 RGB frames and state values, validates the
native hardware feature schema without connecting any transports, and reports
named action dimensions plus neural forward/action selection timings. It runs
31 selections per stage to exercise more than one default H=30 action chunk.
`--benchmark-actions 100` extends this. For a non-commanding check with real
observations, add `--connect-hardware`: the wrapper opens native buses/cameras
without native motor configuration, torque toggling, goal writes or hold writes.
Dry-run predictions are never executed. Synthetic inputs do not test folding
quality, real image preprocessing distributions or real control throughput.

Normal deployment loads and validates both policies before connecting followers,
performs a non-motion inference check, then requires `y` before motion. Native
connection configures follower motors, so support the arms during connection.
At each stage end it discards queued actions and commands a measured-position
hold. Only a successful operator verification advances to the next fold.
`verify: false` permits unverified progression but never reports a successful
fold/task. EOF or failed verification aborts. Each new stage/attempt resets both
ACT and processor state. Another attempt needs explicit confirmation.

The deadline-based loop skips missed slots instead of bursting catch-up actions.
It checks stops/timeouts/duration again after observation and inference, and
rejects an action if inference exceeds `action_max_age_s`. It logs actual action
frequency, overruns, elapsed time, count, model-forward and selection latency.
Synchronous CPU forwards can cause overruns; 30 Hz is a target, not a measured
guarantee. Evaluate on the Framework laptop before approving trajectories.

## Stop behavior and limitations

Normal completion, failed verification, stage timeout, invalid output, camera
fault, motor fault, Ctrl+C, SIGTERM and unexpected exceptions all end policy
execution and discard queues. Holds read each arm independently and command its
current measured position. If one arm cannot communicate, holding the other is
still attempted. Shutdown closes cameras and buses and preserves follower
torque. Hold failures are logged; hardware communication failure can prevent a
software hold from taking effect. An interrupted transport call can delay stop
handling. Operator verification has no automated time limit while arms hold.

There is no automatic home motion or collision planner. Joint/step limits do
not establish bimanual collision avoidance. Physical workspace validation,
conservative settings, safe demonstrations and operator supervision are required.
Use an independent physical motor-power cutoff; software stop cannot replace it.
Support arms before intentionally removing power. CFOLD's file locks prevent
its own record/teleop/deploy commands from sharing ports; unrelated programs
must also be kept away from those devices.

For a single-arm experiment use `robot.type: so101_follower`, move follower
port/torque/safety fields from `left_arm_config` directly onto `robot`, remove
the two arm subconfigs, and leave camera definitions in the top-level `cameras`
mapping (`arm` is then ignored). Use `teleop.type: so101_leader` with its port
directly on `teleop`, and set optional
calibration mappings to `{single: EXISTING_ID}`. Train separate compatible
policies and metadata. The wrapper derives dimensions; it never truncates a
bimanual model's output. The task still has two sequential folds.

## Tests

```bash
python -m pytest -q
# Optional real trained-model test, with all files cached/local:
CFOLD_CHECKPOINT=checkpoints/fold1 CFOLD_METADATA=checkpoints/fold1/dataset_info.json \
  python -m pytest -q tests/test_robot_policy.py::test_trained_checkpoint_inference
```

Tests cover native configuration decoding, schema/name ordering, FSM transitions,
successful and failed attempts, skipped verification, timeouts, blocked-inference
stop/deadline checks, queue resets, safe action rejection, fault cleanup, port
exclusion, and a real ACT/processor save/load/forward/postprocess round trip with
small random weights. The latter tests integration, not a learned folding skill.
The trained-checkpoint test skips unless explicitly supplied. No hardware tests
have been run. Model weights, datasets, generated outputs and logs are ignored.
