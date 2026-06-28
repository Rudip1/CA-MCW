# models/ — trained model artifacts

Each model family has its own subdirectory. Within each family, runs are
organised by name in isolated subdirectories so multiple trained models can
coexist without overwriting each other.

## Naming convention

Run folders follow `<channels>_<runtag>` where:

- `<channels>` is the channel set the model was trained on:
  - `v1` → channels_v1 (126 dims)
  - `v2` → channels_v2 (130 dims)
  - `v3` → channels_v3 (170 dims, full feature set)
- `<runtag>` is free-form. The current canonical hp-group runtags are:
  - `normal`  — default hyperparameters (arch_default; see EXPERIMENTS.md)
  - `tuned`   — moderate regularization (arch_small + dropout 0.3 + wd 1e-3)
  - `hardreg` — aggressive regularization (arch_tiny + dropout 0.5 + wd 1e-2 + batch 64)

Example folder names: `v1_normal`, `v2_tuned`, `v3_hardreg`.

Older runtags (e.g. `v1_batchmanual_pravin_2026_05_11`, `v1_manual_v1`) are
preserved in `_archive_pre_clean_naming_2026_05_11/` for reference but not
consumed by any active launch.

Why bake the channel set into the folder: collected HDF5s always store the
full v3 features, but trainers can slice down to v1 / v2 / v3. The folder
name records which slice the saved weights were trained on, so loading the
wrong-dim ONNX at runtime fails fast instead of silently mis-aligning
features.

## Directory layout

```
models/
  metacritic_raw_wt/          ← raw-critic cost training runs (no oracle QP)
    v1_manual_v1/
      meta_critic.onnx        consumed by metacritic_inference_node.py sidecar
      meta_critic.pt          PyTorch checkpoint
      feature_norm.json       normalisation stats (mean/std per feature dim)
      metadata.json           ONNX metadata (in_dim, channel_names, run_id …)
      training_log.csv        per-epoch loss history
      training_curve.png      loss curve figure
    v3_seed0/
      (same files)
    …

  metacritic_oracle_wt/       ← Phase-9 QP oracle-label training runs
    v1_manual_v1/
      (same files as raw)
    …

  imitation_wt/               ← behaviour cloning (vx, wz direct output)
    v1_manual_v1/
      imitation.onnx          consumed by imitation_inference_node.py sidecar
      imitation.pt
      feature_norm.json
      metadata.json
      training_log.csv
      training_curve.png
    …
```

Binary `.onnx` and `.pt` files are gitignored. `feature_norm.json`,
`metadata.json`, `training_log.csv`, and `training_curve.png` are committed
so the run is reproducible and auditable without re-running training.

## Selecting weights at launch

The active run folder is controlled by `vf_controller/model_defaults.py`
(comment / uncomment one block per family):

```python
DEFAULT_INFERENCE_TYPE    = "oracle"        # "raw" | "oracle"
DEFAULT_INFERENCE_WEIGHTS = "v3_hardreg"    # run folder under metacritic_<type>_wt/
DEFAULT_IMITATION_WEIGHTS = "v3_hardreg"    # run folder under imitation_wt/
```

To switch at launch time without editing the file:

```bash
# Inference: switch to a v2 raw model trained with the tuned hp group
ros2 launch vf_robot_bringup bringup_launch.py \
    inference_model_type:=raw  inference_weights:=v2_tuned

# Imitation: switch to v3 normal
ros2 launch vf_robot_bringup bringup_launch.py \
    controller:=vf_imitationwt  imitation_weights:=v3_normal

# Custom path (escape hatch — bypasses folder resolution)
ros2 launch vf_robot_bringup bringup_launch.py \
    onnx_path:=/abs/path/to/meta_critic.onnx
```

For closed-loop batch evaluation, use the per-family launches under
`vf_robot_utils/launch/vf_data_evaluation/batch/`:

```bash
ros2 launch vf_robot_utils vf_data_evaluation_batch_rawwt.launch.py \
    map:=house_my1_map hp:=hardreg ch:=v3 planners:=NavFn
```

See `EVAL_SWEEP.md` for the full 27-variant sweep procedure.

## Training scripts and output folders

| Script | Default `--parent-dir` | Output `<run-name>` shape |
|--------|------------------------|---------------------------|
| `train_raw_critics.py` | `models/metacritic_raw_wt/`    | `<channels>_<runtag>` |
| `train_inference.py`   | `models/metacritic_oracle_wt/` | `<channels>_<runtag>` |
| `train_imitation.py`   | `models/imitation_wt/`         | `<channels>_<runtag>` |

Pass `--channels v1|v2|v3` and `--run-name <runtag>` (the trainer prefixes
the channels). Each script fails immediately if the run folder already
exists — no silent overwrites.

## ONNX I/O signatures

| Model | Input name | Input shape | Output name | Output shape |
|-------|------------|-------------|-------------|--------------|
| `meta_critic.onnx` | `features` | `(1, D)` | `weights` | `(1, K)` |
| `imitation.onnx`   | `features` | `(1, D)` | `cmd_vel`  | `(1, 2)` |

`D` matches the `<channels>` prefix (126 / 130 / 170); `K` is the number
of critics (default 11 for vf_fixedwt.yaml).
