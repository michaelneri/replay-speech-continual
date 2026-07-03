#!/usr/bin/env python3
"""
training_cl_algorithms.py — CL algorithm training for multi-channel audio anti-spoofing.

Set ALGORITHM at the top to select the method:
  "ewc"  — Elastic Weight Consolidation  (regularization-based)
  "gpm"  — Gradient Projection Memory    (optimization-based)
  "tsb"  — Task-Specific Beamformer      (architecture-based)

Protocol — same domain-incremental setup as train_cl.py
─────────────────────────────────────────────────────────
All 4! orderings × NUM_RUNS runs, prefix-cached CL fine-tuning,
plus single_ref and joint_ref for FWT / IM metrics.

Algorithm-specific behaviour
─────────────────────────────
EWC / GPM:
  After every CL training step k, consolidation state (.ewc.pt / .gpm.pt)
  is written next to the checkpoint.  Fine-tuning at step k+1 loads that
  state so the penalty / projection targets the correct previous task.

TSB:
  _build_model() receives task_idx = len(prefix) - 1 so that only the
  k-th beamformer is trainable.  Old beamformers are frozen automatically
  by activate_task().  At evaluation the ensemble of all trained heads is
  used (no task label needed).
"""

import json
import os
import time
from datetime import datetime
from itertools import combinations, permutations
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
import torchvision.transforms.v2 as tv_transforms
import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from Remasc import (
    AdjustChannel, AdjustLength, MIC,
    NUM_CHANNELS_MIC, RemascDataModule,
)
from cl_metrics import compute_cl_metrics
from model_cl import (
    EWCModelModule, GPMModelModule, TSBModelModule,
    consolidate_ewc, consolidate_gpm,
)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Algorithm selector  ← change this to "ewc", "gpm", or "tsb"
# ─────────────────────────────────────────────────────────────────────────────
ALGORITHM: str = "gpm"   # "ewc" | "gpm" | "tsb"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CFG: dict = {
    "rdevice"       : 2,
    "orig_fs"       : 44_100,
    "target_fs"     : 16_000,
    "n_fft"         : 1024,
    "hop_length"    : 512,
    "seconds"       : 1,
    "batch_size"    : 8,
    "lr"            : 1e-4,
    "max_epochs"    : 50,
    "num_runs"      : 5,
    "envs"          : [1, 2, 3, 4],
    "base_seed"     : 30,
    # EWC-specific
    "ewc_lambda"    : 5_000.0,
    # GPM-specific
    "gpm_threshold" : 0.97,
}

INPUT_CH  = NUM_CHANNELS_MIC[MIC[CFG["rdevice"]]]
AUDIO_LEN = int(CFG["target_fs"] * CFG["seconds"])
ENVS      = CFG["envs"]

CKPT_ROOT      = Path("cl_checkpoints") / f"D{CFG['rdevice']}_{CFG['target_fs']}Hz_{ALGORITHM}"
RESULTS        = Path(f"cl_results_{ALGORITHM}.json")
EER_CACHE_PATH = CKPT_ROOT / "eer_cache.json"

ALL_ORDERS = list(permutations(ENVS))

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────
def _run_seed(run: int) -> int:
    return CFG["base_seed"] + run * 1_000


def _save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _find_best_ckpt(directory: Path) -> Optional[str]:
    hits = sorted(directory.glob("best-*.ckpt"))
    return str(hits[-1].resolve()) if hits else None


def _prefix_to_str(prefix: tuple) -> str:
    return "_".join(map(str, prefix))


def _order_to_str(order: tuple) -> str:
    return "_".join(map(str, order))


# ─────────────────────────────────────────────────────────────────────────────
# Transform pipeline
# ─────────────────────────────────────────────────────────────────────────────
class _Resample(torch.nn.Module):
    def __init__(self, orig_freq: int, new_freq: int) -> None:
        super().__init__()
        self._rs   = torchaudio.transforms.Resample(orig_freq, new_freq)
        self._freq = new_freq

    def forward(self, sample: dict) -> dict:
        sample["waveform"]    = self._rs(sample["waveform"])
        sample["sample_rate"] = self._freq
        return sample


class _SafeNormScale(torch.nn.Module):
    def __init__(self, scale: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale = scale
        self.eps   = eps

    def forward(self, sample: dict) -> dict:
        w = sample["waveform"]
        w = torch.nan_to_num(w, nan=0.0, posinf=1.0, neginf=-1.0)
        max_amp = w.abs().max().item()
        if max_amp > self.eps:
            w = (w / max_amp) * self.scale
        sample["waveform"] = w
        return sample


def make_transform() -> tv_transforms.Compose:
    return tv_transforms.Compose([
        AdjustChannel(INPUT_CH),
        _Resample(CFG["orig_fs"], CFG["target_fs"]),
        AdjustLength(AUDIO_LEN),
        _SafeNormScale(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Model construction helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_module_class():
    return {"ewc": EWCModelModule, "gpm": GPMModelModule, "tsb": TSBModelModule}[ALGORITHM]


def _build_model(
    label_0:  int,
    label_1:  int,
    tag:      str,
    task_idx: int = 0,
):
    common = dict(
        input_channels = INPUT_CH,
        lr             = CFG["lr"],
        train_label_0  = label_0,
        train_label_1  = label_1,
        n_fft          = CFG["n_fft"],
        hop_length     = CFG["hop_length"],
        fs             = CFG["target_fs"],
        info           = tag,
    )
    if ALGORITHM == "ewc":
        return EWCModelModule(**common, ewc_lambda=CFG["ewc_lambda"])
    elif ALGORITHM == "gpm":
        return GPMModelModule(**common, gpm_threshold=CFG["gpm_threshold"])
    elif ALGORITHM == "tsb":
        return TSBModelModule(**common, task_idx=task_idx, num_tasks=len(ENVS))
    else:
        raise ValueError(f"Unknown ALGORITHM: {ALGORITHM!r}")


def _load_weights_into(model, ckpt_path: str) -> None:
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"])


def _consolidate_after_training(ckpt_path: str, train_env: int) -> None:
    """Write EWC or GPM consolidation state next to the checkpoint."""
    if ALGORITHM not in ("ewc", "gpm"):
        return
    ext = ".ewc.pt" if ALGORITHM == "ewc" else ".gpm.pt"
    if os.path.exists(ckpt_path + ext):
        return
    tf = make_transform()
    dm = RemascDataModule(
        batch_size=CFG["batch_size"], env=[train_env],
        rdevice=CFG["rdevice"], num_workers=0, transform=tf,
    )
    consolidate_fn = consolidate_ewc if ALGORITHM == "ewc" else consolidate_gpm
    print(f"    [consolidate-{ALGORITHM}] env={train_env}  {ckpt_path}")
    consolidate_fn(ckpt_path, dm.train_dataloader(), device=_DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_one(
    train_envs:              list[int],
    val_envs:                list[int],
    ckpt_dir:                Path,
    run:                     int,
    tag:                     str,
    wandb_tags:              list[str],
    load_weights_from:       Optional[str] = None,
    task_idx:                int           = 0,
    load_consolidation_from: Optional[str] = None,
) -> str:
    """
    Train (or fine-tune) the model, save best checkpoint, return its path.

    load_consolidation_from : for EWC / GPM — load Fisher / gradient bases
                              from this checkpoint's sidecar file before training.
    task_idx                : for TSB — which beamformer head to train.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(_run_seed(run), workers=True)

    tf = make_transform()

    train_dm = RemascDataModule(
        batch_size=CFG["batch_size"], env=train_envs,
        rdevice=CFG["rdevice"], num_workers=0, transform=tf,
    )
    val_dm = RemascDataModule(
        batch_size=CFG["batch_size"], env=val_envs,
        rdevice=CFG["rdevice"], num_workers=0, transform=tf,
    )
    label_0, label_1 = train_dm.count_genuine_and_replay()

    model = _build_model(int(label_0), int(label_1), tag, task_idx=task_idx)
    if load_weights_from is not None:
        _load_weights_into(model, load_weights_from)
    if load_consolidation_from is not None and ALGORITHM in ("ewc", "gpm"):
        model.load_consolidation(load_consolidation_from)

    cb = ModelCheckpoint(
        dirpath                 = str(ckpt_dir),
        filename                = "best-{val_eer:.4f}",
        monitor                 = "val_eer",
        mode                    = "min",
        save_top_k              = 1,
        save_last               = False,
        every_n_epochs          = 1,
        auto_insert_metric_name = False,
    )
    ts     = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = WandbLogger(
        project  = "ReplaySpeechCL",
        name     = f"{tag}_{ts}",
        tags     = wandb_tags,
        save_dir = str(ckpt_dir),
    )

    trainer = Trainer(
        devices              = 1,
        max_epochs           = CFG["max_epochs"],
        callbacks            = [cb],
        logger               = logger,
        num_sanity_val_steps = 0,
        deterministic        = "warn",
        enable_progress_bar  = True,
        gradient_clip_val    = 1.0,
    )
    torch.set_float32_matmul_precision("medium")

    trainer.fit(
        model,
        train_dataloaders = train_dm.train_dataloader(),
        val_dataloaders   = val_dm.val_dataloader(),
    )
    wandb.finish()
    return cb.best_model_path


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_eer(
    ckpt_path: str,
    test_env:  int,
    run:       int,
    cache:     dict,
) -> float:
    key = f"ckpt={ckpt_path}|env={test_env}|run={run}"
    if key in cache:
        return float(cache[key])

    seed_everything(_run_seed(run), workers=True)
    tf = make_transform()

    dm = RemascDataModule(
        batch_size=CFG["batch_size"], env=[test_env],
        rdevice=CFG["rdevice"], num_workers=0, transform=tf,
    )
    model = _get_module_class().load_from_checkpoint(ckpt_path)

    trainer = Trainer(
        devices             = 1,
        logger              = False,
        enable_progress_bar = False,
        deterministic       = "warn",
    )
    res = trainer.test(model, dataloaders=dm.test_dataloader(), verbose=False)
    eer = float(res[0]["test_err"])

    cache[key] = eer
    save_eer_cache(cache)
    return eer


def save_eer_cache(cache: dict) -> None:
    _save_json(cache, EER_CACHE_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — single-environment reference models
# ─────────────────────────────────────────────────────────────────────────────
def phase1_single_ref(
    run:   int,
    cache: dict,
) -> dict[int, str]:
    env_to_ckpt: dict[int, str] = {}

    for env in ENVS:
        ckpt_dir = CKPT_ROOT / f"run_{run}" / "single_ref" / f"env_{env}"
        existing = _find_best_ckpt(ckpt_dir)

        if existing:
            print(f"    [skip] single_ref env={env} run={run}  — {existing}")
            env_to_ckpt[env] = existing
        else:
            print(f"    [train] single_ref env={env} run={run}")
            ckpt = train_one(
                train_envs=[env], val_envs=[env],
                ckpt_dir=ckpt_dir, run=run,
                tag=f"single_ref_e{env}_r{run}_{ALGORITHM}",
                wandb_tags=[
                    f"D{CFG['rdevice']}", f"{CFG['target_fs']}Hz",
                    f"n_fft{CFG['n_fft']}", "single_ref",
                    f"env{env}", f"run{run}", ALGORITHM,
                ],
                task_idx=0,
            )
            env_to_ckpt[env] = ckpt

        # Consolidate step-0 model so step-1 can load its state
        _consolidate_after_training(env_to_ckpt[env], train_env=env)

        eer = evaluate_eer(env_to_ckpt[env], test_env=env, run=run, cache=cache)
        print(f"      EER[env={env}] = {eer:.4f}")

    return env_to_ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — joint reference models
# ─────────────────────────────────────────────────────────────────────────────
def phase2_joint_ref(
    run:             int,
    single_ref_ckpt: dict[int, str],
    cache:           dict,
) -> dict[frozenset, str]:
    fset_to_ckpt: dict[frozenset, str] = {
        frozenset({e}): single_ref_ckpt[e] for e in ENVS
    }

    for size in range(2, len(ENVS) + 1):
        for subset in combinations(ENVS, size):
            fset     = frozenset(subset)
            sorted_s = sorted(subset)
            env_str  = "_".join(map(str, sorted_s))
            ckpt_dir = CKPT_ROOT / f"run_{run}" / "joint" / f"envs_{env_str}"
            existing = _find_best_ckpt(ckpt_dir)

            if existing:
                print(f"    [skip] joint envs={env_str} run={run}  — {existing}")
                fset_to_ckpt[fset] = existing
            else:
                print(f"    [train] joint envs={env_str} run={run}")
                ckpt = train_one(
                    train_envs=list(subset), val_envs=list(subset),
                    ckpt_dir=ckpt_dir, run=run,
                    tag=f"joint_envs{env_str}_r{run}_{ALGORITHM}",
                    wandb_tags=[
                        f"D{CFG['rdevice']}", f"{CFG['target_fs']}Hz",
                        f"n_fft{CFG['n_fft']}", "joint_ref",
                        f"envs{env_str}", f"run{run}", ALGORITHM,
                    ],
                    task_idx=0,
                )
                fset_to_ckpt[fset] = ckpt

            for e in subset:
                eer = evaluate_eer(fset_to_ckpt[fset], test_env=e, run=run, cache=cache)
                print(f"      EER[envs={env_str}, test_env={e}] = {eer:.4f}")

    return fset_to_ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — CL fine-tuning with prefix caching
# ─────────────────────────────────────────────────────────────────────────────
def _get_or_train_prefix(
    run:             int,
    prefix:          tuple,
    single_ref_ckpt: dict[int, str],
    prefix_cache:    dict[tuple, str],
) -> str:
    if prefix in prefix_cache:
        return prefix_cache[prefix]

    if len(prefix) == 1:
        # Step 0: single_ref model (already consolidated in phase1_single_ref)
        ckpt = single_ref_ckpt[prefix[0]]
        prefix_cache[prefix] = ckpt
        return ckpt

    prev_prefix = prefix[:-1]
    prev_ckpt   = _get_or_train_prefix(run, prev_prefix, single_ref_ckpt, prefix_cache)

    new_env  = prefix[-1]
    task_idx = len(prefix) - 1   # used by TSB
    ckpt_dir = CKPT_ROOT / f"run_{run}" / "cl_prefix" / _prefix_to_str(prefix)
    existing = _find_best_ckpt(ckpt_dir)

    if existing:
        print(f"    [skip] CL prefix={prefix} run={run}  — {existing}")
        ckpt = existing
    else:
        print(f"    [fine-tune] CL prefix={prefix} run={run}  (new_env={new_env}, {ALGORITHM})")
        ckpt = train_one(
            train_envs=[new_env], val_envs=[new_env],
            ckpt_dir=ckpt_dir, run=run,
            tag=f"cl_{ALGORITHM}_prefix{'_'.join(map(str,prefix))}_r{run}",
            wandb_tags=[
                f"D{CFG['rdevice']}", f"{CFG['target_fs']}Hz",
                f"n_fft{CFG['n_fft']}", "cl_finetune",
                f"step{len(prefix)-1}", f"new_env{new_env}",
                f"prefix{'_'.join(map(str,prefix))}", f"run{run}", ALGORITHM,
            ],
            load_weights_from=prev_ckpt,
            task_idx=task_idx,
            load_consolidation_from=prev_ckpt,
        )

    # Consolidate so the next step can load state (EWC/GPM only)
    _consolidate_after_training(ckpt, train_env=new_env)

    prefix_cache[prefix] = ckpt
    return ckpt


def phase3_cl(
    run:             int,
    single_ref_ckpt: dict[int, str],
    fset_to_ckpt:    dict[frozenset, str],
    cache:           dict,
) -> dict:
    prefix_cache: dict[tuple, str] = {}
    results: dict = {}

    for order_idx, order in enumerate(ALL_ORDERS):
        order_str = _order_to_str(order)
        print(f"\n  Order {order_idx+1}/{len(ALL_ORDERS)}: {order}")
        results[order_str] = {}

        for k in range(len(ENVS)):
            prefix = order[: k + 1]
            ckpt   = _get_or_train_prefix(run, prefix, single_ref_ckpt, prefix_cache)

            step_eers: dict[int, float] = {}
            for test_env in ENVS:
                eer = evaluate_eer(ckpt, test_env=test_env, run=run, cache=cache)
                step_eers[test_env] = eer
                print(f"      step {k}, test_env={test_env}: EER={eer:.4f}")

            results[order_str][k] = step_eers

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CL metrics computation
# ─────────────────────────────────────────────────────────────────────────────
def build_metrics(
    order:           tuple,
    run_results:     dict,
    single_ref_ckpt: dict[int, str],
    fset_to_ckpt:    dict[frozenset, str],
    run:             int,
    cache:           dict,
) -> dict:
    K         = len(ENVS)
    order_str = _order_to_str(order)
    step_data = run_results[order_str]

    perf_matrix = np.full((K, K), np.nan)
    for k in range(K):
        for j in range(K):
            perf_matrix[k, j] = step_data[k][order[j]]

    joint_ref_arr = np.zeros(K)
    for k in range(K):
        fset = frozenset(order[: k + 1])
        ckpt = fset_to_ckpt[fset]
        joint_ref_arr[k] = evaluate_eer(ckpt, test_env=order[k], run=run, cache=cache)

    single_ref_arr = np.array([
        evaluate_eer(single_ref_ckpt[order[j]], test_env=order[j], run=run, cache=cache)
        for j in range(K)
    ])

    return compute_cl_metrics(
        perf_matrix,
        joint_ref       = joint_ref_arr,
        single_ref      = single_ref_arr,
        lower_is_better = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    wandb.login()
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    cache = _load_json(EER_CACHE_PATH)

    num_single   = len(ENVS)
    num_joint    = sum(1 for s in range(2, len(ENVS)+1)
                       for _ in combinations(ENVS, s))
    num_cl_steps = (12 + 24 + 24) * CFG["num_runs"]
    total_trains = (num_single + num_joint) * CFG["num_runs"] + num_cl_steps
    print("=" * 65)
    print(f"CL Algorithm: {ALGORITHM.upper()}  |  Device {CFG['rdevice']}  |  "
          f"{CFG['target_fs']} Hz  |  n_fft={CFG['n_fft']}")
    print(f"Orders: {len(ALL_ORDERS)}  ×  Runs: {CFG['num_runs']}  =  "
          f"{len(ALL_ORDERS)*CFG['num_runs']} CL experiments")
    print(f"Estimated training calls: {total_trains}  "
          f"({total_trains*CFG['max_epochs']} epochs total)")
    print("=" * 65)

    raw_results: dict = {}
    per_order_metrics: dict[str, dict[str, list]] = {}
    for order in ALL_ORDERS:
        key = _order_to_str(order)
        per_order_metrics[key] = {m: [] for m in ("AA","AIA","FM","BWT","IM","FWT")}

    for run in range(CFG["num_runs"]):
        print(f"\n{'━'*65}")
        print(f"  RUN  {run + 1} / {CFG['num_runs']}"
              f"   (seed = {_run_seed(run)})")
        print(f"{'━'*65}")

        print("\n[Phase 1] Single-environment reference models")
        single_ref_ckpt = phase1_single_ref(run=run, cache=cache)

        print("\n[Phase 2] Joint reference models")
        fset_to_ckpt = phase2_joint_ref(
            run=run, single_ref_ckpt=single_ref_ckpt, cache=cache
        )

        print("\n[Phase 3] CL fine-tuning (all orderings, prefix-cached)")
        run_results = phase3_cl(
            run=run,
            single_ref_ckpt=single_ref_ckpt,
            fset_to_ckpt=fset_to_ckpt,
            cache=cache,
        )
        raw_results[run] = run_results

        for order in ALL_ORDERS:
            m = build_metrics(
                order=order,
                run_results=run_results,
                single_ref_ckpt=single_ref_ckpt,
                fset_to_ckpt=fset_to_ckpt,
                run=run,
                cache=cache,
            )
            ostr = _order_to_str(order)
            for key in ("AA", "AIA", "FM", "BWT", "IM", "FWT"):
                val = m[key]
                if val is not None:
                    per_order_metrics[ostr][key].append(float(val))

    print("\n\n" + "=" * 65)
    print(f"RESULTS [{ALGORITHM.upper()}]  —  mean ± std  (across 5 runs × 24 orderings)")
    print("=" * 65)

    global_vals: dict[str, list] = {m: [] for m in ("AA","AIA","FM","BWT","IM","FWT")}
    per_order_summary: dict = {}

    for order in ALL_ORDERS:
        ostr = _order_to_str(order)
        per_order_summary[ostr] = {}
        for metric, vals in per_order_metrics[ostr].items():
            if vals:
                arr = np.array(vals)
                per_order_summary[ostr][metric] = {
                    "mean"   : float(arr.mean()),
                    "std"    : float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
                    "values" : vals,
                }
                global_vals[metric].extend(vals)
            else:
                per_order_summary[ostr][metric] = None

    global_summary: dict = {}
    for metric, vals in global_vals.items():
        if vals:
            arr = np.array(vals)
            global_summary[metric] = {
                "mean" : float(arr.mean()),
                "std"  : float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
                "n"    : len(arr),
            }
            direction = ("(lower=better)" if metric in ("AA","AIA")
                         else "(>0=forgot/intrans.)" if metric in ("FM","IM")
                         else "(<0=forgot)" if metric == "BWT"
                         else "(>0=pos.transfer)")
            print(f"  {metric:<5}  {global_summary[metric]['mean']:+.4f} ± "
                  f"{global_summary[metric]['std']:.4f}  {direction}")

    print(f"\n  Total wall-clock: {(time.time()-t0)/3600:.2f} h")
    print("=" * 65)

    output = {
        "algorithm"         : ALGORITHM,
        "config"            : CFG,
        "global_summary"    : global_summary,
        "per_order_summary" : per_order_summary,
        "raw_perf_matrices" : {
            f"run{r}_{_order_to_str(o)}": {
                f"step{k}": {str(e): raw_results[r][_order_to_str(o)][k][e]
                              for e in ENVS}
                for k in range(len(ENVS))
            }
            for r in range(CFG["num_runs"])
            for o in ALL_ORDERS
        },
        "elapsed_hours"     : (time.time() - t0) / 3600,
    }
    _save_json(output, RESULTS)
    print(f"\nFull results saved → {RESULTS}")


if __name__ == "__main__":
    main()
