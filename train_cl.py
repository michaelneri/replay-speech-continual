#!/usr/bin/env python3
"""
train_cl.py — Naive fine-tuning (catastrophic forgetting) baseline for
              multi-channel audio anti-spoofing on ReMASC.

Protocol — Domain-Incremental Learning on environments 1-4, Device 2
─────────────────────────────────────────────────────────────────────
For each ordering  [A, B, C, D]  of the four environments and each of
NUM_RUNS independent runs:

  step 0  train from scratch on A;      test on [A, B, C, D]
  step 1  fine-tune on B (from step 0); test on [A, B, C, D]
  step 2  fine-tune on C (from step 1); test on [A, B, C, D]
  step 3  fine-tune on D (from step 2); test on [A, B, C, D]

This gives the 4×4 EER matrix  perf[k, j] needed for CL metrics.

Reference models (run once per run, reused across orderings):
  single_ref[e]   — trained only on env e   (needed for FWT)
  joint_ref[k]    — trained jointly on {A,…,e_k}  (needed for IM)

Prefix caching — model trained on prefix (A,) is the same for ALL
orderings that start with A, so it is trained only once per run.
Unique CL training calls per run: 4 (len-1) + 12 + 24 + 24 = 64,
but len-1 = single_ref so net new: 60 fine-tuning calls per run.

Configuration
─────────────
  Device      : 3  (RES_CORE, 6 ch)
  Sample rate : 44 100 → resampled to 16 000 Hz
  n_fft       : 1024,  hop_length : 512
  Duration    : 1 second (16 000 samples)
  Orderings   : all 4! = 24
  Runs        : 5 (for 95 % confidence intervals)

Checkpoint layout
─────────────────
cl_checkpoints/D2_16kHz/
  eer_cache.json                         ← persisted test-EER cache
  run_<r>/
    single_ref/env_<e>/best-*.ckpt       ← single-env reference
    joint/envs_<sorted>/best-*.ckpt      ← joint reference (size ≥ 2)
    cl_prefix/<e0>_…_<ek>/best-*.ckpt   ← prefix model after step k

Results
───────
cl_results.json — per-ordering metrics + overall statistics
"""

import json
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
from model import AudioDeepFakeDetectionModelModule
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CFG: dict = {
    "rdevice"    : 2,
    "orig_fs"    : 44_100,      # device 2 native sample rate
    "target_fs"  : 16_000,      # downsample target
    "n_fft"      : 1024,
    "hop_length" : 512,
    "seconds"    : 1,
    "batch_size" : 8,
    "lr"         : 1e-4,
    "max_epochs" : 50,
    "num_runs"   : 5,
    "envs"       : [1, 2, 3, 4],
    "base_seed"  : 30,
}

INPUT_CH  = NUM_CHANNELS_MIC[MIC[CFG["rdevice"]]]          # 6 channels
AUDIO_LEN = int(CFG["target_fs"] * CFG["seconds"])          # 16 000 samples
ENVS      = CFG["envs"]

CKPT_ROOT  = Path("cl_checkpoints") / f"D{CFG['rdevice']}_{CFG['target_fs']}Hz"
RESULTS    = Path("cl_results.json")
EER_CACHE_PATH = CKPT_ROOT / "eer_cache.json"

ALL_ORDERS = list(permutations(ENVS))   # 24 orderings


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
    """Return the newest best-*.ckpt in *directory*, or None."""
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
    """Resample waveform and update sample_rate key in the sample dict."""

    def __init__(self, orig_freq: int, new_freq: int) -> None:
        super().__init__()
        self._rs   = torchaudio.transforms.Resample(orig_freq, new_freq)
        self._freq = new_freq

    def forward(self, sample: dict) -> dict:
        sample["waveform"]     = self._rs(sample["waveform"])
        sample["sample_rate"]  = self._freq
        return sample


class _SafeNormScale(torch.nn.Module):
    """
    Normalise peak amplitude to 1, but skip normalisation if the waveform
    is silent (max amplitude ≤ eps).  The original NormScale divides by the
    exact max, which produces inf/NaN for all-zero frames that can occur
    after resampling short clips — causing NaN loss from the very first
    affected batch.
    """

    def __init__(self, scale: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale = scale
        self.eps   = eps

    def forward(self, sample: dict) -> dict:
        w = sample["waveform"]
        # First sanitize any NaN/Inf that may come from corrupted audio or resampling
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
        _SafeNormScale(),          # NaN-safe replacement for NormScale
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Model construction helpers
# ─────────────────────────────────────────────────────────────────────────────
def _build_model(
    label_0: int,
    label_1: int,
    tag: str,
) -> AudioDeepFakeDetectionModelModule:
    return AudioDeepFakeDetectionModelModule(
        input_channels = INPUT_CH,
        lr             = CFG["lr"],
        train_label_0  = label_0,
        train_label_1  = label_1,
        n_fft          = CFG["n_fft"],
        hop_length     = CFG["hop_length"],
        fs             = CFG["target_fs"],
        info           = tag,
    )


def _load_weights_into(model: AudioDeepFakeDetectionModelModule,
                        ckpt_path: str) -> None:
    """Copy only the neural-network weights; optimizer stays fresh."""
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"])


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_one(
    train_envs: list[int],
    val_envs:   list[int],
    ckpt_dir:   Path,
    run:        int,
    tag:        str,
    wandb_tags: list[str],
    load_weights_from: Optional[str] = None,
) -> str:
    """
    Train (or fine-tune) the model, save best checkpoint, return its path.

    Parameters
    ----------
    train_envs         : environments used for training (core split)
    val_envs           : environments used for validation (eval split)
    ckpt_dir           : where to write checkpoints and W&B logs
    run                : run index — used for seeding
    tag                : W&B run name (human-readable, datetime appended)
    wandb_tags         : list of categorical tags for W&B
    load_weights_from  : if not None, load ONLY the neural-network weights
                         from this checkpoint (optimizer starts fresh)
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

    model = _build_model(int(label_0), int(label_1), tag)
    if load_weights_from is not None:
        _load_weights_into(model, load_weights_from)

    cb = ModelCheckpoint(
        dirpath               = str(ckpt_dir),
        filename              = "best-{val_eer:.4f}",
        monitor               = "val_eer",
        mode                  = "min",
        save_top_k            = 1,
        save_last             = False,
        every_n_epochs        = 1,
        auto_insert_metric_name = False,
    )
    ts     = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = WandbLogger(
        project = "ReplaySpeechCL",
        name    = f"{tag}_{ts}",
        tags    = wandb_tags,
        save_dir = str(ckpt_dir),
    )

    trainer = Trainer(
        devices              = 1,
        max_epochs           = CFG["max_epochs"],
        callbacks            = [cb],
        logger               = logger,
        num_sanity_val_steps = 0,
        deterministic        = "warn",   # GRU on CUDA is not fully deterministic
        enable_progress_bar  = True,
        gradient_clip_val    = 1.0,      # prevent NaN gradients from bad batches
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
# Evaluation (EER on a single environment, with disk cache)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_eer(
    ckpt_path: str,
    test_env:  int,
    run:       int,
    cache:     dict,
) -> float:
    """
    Return EER (%) on *test_env* for the model in *ckpt_path*.
    Results are memoised in *cache* (written to disk after each new call).
    """
    key = f"ckpt={ckpt_path}|env={test_env}|run={run}"
    if key in cache:
        return float(cache[key])

    seed_everything(_run_seed(run), workers=True)
    tf = make_transform()

    dm = RemascDataModule(
        batch_size=CFG["batch_size"], env=[test_env],
        rdevice=CFG["rdevice"], num_workers=0, transform=tf,
    )
    model = AudioDeepFakeDetectionModelModule.load_from_checkpoint(ckpt_path)

    trainer = Trainer(
        devices             = 1,
        logger              = False,
        enable_progress_bar = False,
        deterministic       = "warn",
    )
    res  = trainer.test(model, dataloaders=dm.test_dataloader(), verbose=False)
    eer  = float(res[0]["test_err"])

    cache[key] = eer
    save_eer_cache(cache)
    return eer


def save_eer_cache(cache: dict) -> None:
    _save_json(cache, EER_CACHE_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — single-environment reference models  (for FWT / single_ref)
# ─────────────────────────────────────────────────────────────────────────────
def phase1_single_ref(
    run:   int,
    cache: dict,
) -> dict[int, str]:
    """
    Train one model per environment.  Returns {env: ckpt_path}.

    These models double as the CL step-0 model for any ordering
    that begins with the respective environment.
    """
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
                tag=f"single_ref_e{env}_r{run}",
                wandb_tags=[
                    f"D{CFG['rdevice']}", f"{CFG['target_fs']}Hz",
                    f"n_fft{CFG['n_fft']}", "single_ref",
                    f"env{env}", f"run{run}", "CL_baseline",
                ],
            )
            env_to_ckpt[env] = ckpt

        # Evaluate on its own environment (for single_ref array)
        eer = evaluate_eer(env_to_ckpt[env], test_env=env, run=run, cache=cache)
        print(f"      EER[env={env}] = {eer:.4f}")

    return env_to_ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — joint reference models  (for IM / joint_ref)
# ─────────────────────────────────────────────────────────────────────────────
def phase2_joint_ref(
    run:            int,
    single_ref_ckpt: dict[int, str],
    cache:          dict,
) -> dict[frozenset, str]:
    """
    Train one joint model for every environment subset of size ≥ 2.
    Size-1 subsets are covered by single_ref.
    Returns {frozenset(subset): ckpt_path}.
    """
    fset_to_ckpt: dict[frozenset, str] = {
        frozenset({e}): single_ref_ckpt[e] for e in ENVS
    }

    for size in range(2, len(ENVS) + 1):
        for subset in combinations(ENVS, size):
            fset      = frozenset(subset)
            sorted_s  = sorted(subset)
            env_str   = "_".join(map(str, sorted_s))
            ckpt_dir  = CKPT_ROOT / f"run_{run}" / "joint" / f"envs_{env_str}"
            existing  = _find_best_ckpt(ckpt_dir)

            if existing:
                print(f"    [skip] joint envs={env_str} run={run}  — {existing}")
                fset_to_ckpt[fset] = existing
            else:
                print(f"    [train] joint envs={env_str} run={run}")
                ckpt = train_one(
                    train_envs=list(subset), val_envs=list(subset),
                    ckpt_dir=ckpt_dir, run=run,
                    tag=f"joint_envs{env_str}_r{run}",
                    wandb_tags=[
                        f"D{CFG['rdevice']}", f"{CFG['target_fs']}Hz",
                        f"n_fft{CFG['n_fft']}", "joint_ref",
                        f"envs{env_str}", f"run{run}", "CL_baseline",
                    ],
                )
                fset_to_ckpt[fset] = ckpt

            # Evaluate on each env in subset (needed for joint_ref lookup)
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
    """
    Recursively ensure the CL model for *prefix* is trained.
    Returns its best checkpoint path.

    prefix = (e0,)          → reuse single_ref[e0]  (no new training)
    prefix = (e0, e1, ...)  → fine-tune from prefix[:-1] on prefix[-1]
    """
    if prefix in prefix_cache:
        return prefix_cache[prefix]

    if len(prefix) == 1:
        # Step 0: the model is the single_ref for that environment.
        ckpt = single_ref_ckpt[prefix[0]]
        prefix_cache[prefix] = ckpt
        return ckpt

    # Ensure predecessor is trained
    prev_prefix = prefix[:-1]
    prev_ckpt   = _get_or_train_prefix(run, prev_prefix, single_ref_ckpt, prefix_cache)

    new_env  = prefix[-1]
    ckpt_dir = CKPT_ROOT / f"run_{run}" / "cl_prefix" / _prefix_to_str(prefix)
    existing = _find_best_ckpt(ckpt_dir)

    if existing:
        print(f"    [skip] CL prefix={prefix} run={run}  — {existing}")
        ckpt = existing
    else:
        print(f"    [fine-tune] CL prefix={prefix} run={run}  (new_env={new_env})")
        ckpt = train_one(
            train_envs=[new_env], val_envs=[new_env],
            ckpt_dir=ckpt_dir, run=run,
            tag=f"cl_prefix{'_'.join(map(str,prefix))}_r{run}",
            wandb_tags=[
                f"D{CFG['rdevice']}", f"{CFG['target_fs']}Hz",
                f"n_fft{CFG['n_fft']}", "cl_finetune",
                f"step{len(prefix)-1}", f"new_env{new_env}",
                f"prefix{'_'.join(map(str,prefix))}", f"run{run}", "CL_baseline",
            ],
            load_weights_from=prev_ckpt,
        )

    prefix_cache[prefix] = ckpt
    return ckpt


def phase3_cl(
    run:             int,
    single_ref_ckpt: dict[int, str],
    fset_to_ckpt:    dict[frozenset, str],
    cache:           dict,
) -> dict:
    """
    For every ordering and every step, collect EER on all 4 environments.

    Returns
    -------
    results : dict  {order_str: {k: {env: eer}}}
    """
    prefix_cache: dict[tuple, str] = {}     # {prefix_tuple: ckpt_path}
    results: dict = {}

    for order_idx, order in enumerate(ALL_ORDERS):
        order_str = _order_to_str(order)
        print(f"\n  Order {order_idx+1}/{len(ALL_ORDERS)}: {order}")
        results[order_str] = {}

        for k in range(len(ENVS)):
            prefix = order[: k + 1]
            ckpt   = _get_or_train_prefix(run, prefix, single_ref_ckpt, prefix_cache)

            # Test on ALL environments
            step_eers: dict[int, float] = {}
            for test_env in ENVS:
                eer = evaluate_eer(ckpt, test_env=test_env, run=run, cache=cache)
                step_eers[test_env] = eer
                print(f"      step {k}, test_env={test_env}: EER={eer:.4f}")

            results[order_str][k] = step_eers

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CL metrics computation as in the paper
# ─────────────────────────────────────────────────────────────────────────────
def build_metrics(
    order:           tuple,
    run_results:     dict,           # {order_str: {k: {env: eer}}}
    single_ref_ckpt: dict[int, str],
    fset_to_ckpt:    dict[frozenset, str],
    run:             int,
    cache:           dict,
) -> dict:
    """
    Build perf_matrix, joint_ref, single_ref arrays and compute CL metrics.

    Returns the metrics dict from compute_cl_metrics.
    """
    K         = len(ENVS)
    order_str = _order_to_str(order)
    step_data = run_results[order_str]

    # perf_matrix[k, j] = EER on order[j] after training on order[0..k]
    perf_matrix = np.full((K, K), np.nan)
    for k in range(K):
        for j in range(K):
            perf_matrix[k, j] = step_data[k][order[j]]

    # joint_ref[k] = EER of joint model on env order[k]
    joint_ref_arr = np.zeros(K)
    for k in range(K):
        fset = frozenset(order[: k + 1])
        ckpt = fset_to_ckpt[fset]
        joint_ref_arr[k] = evaluate_eer(ckpt, test_env=order[k], run=run, cache=cache)

    # single_ref[j] = EER of model trained only on order[j]
    single_ref_arr = np.array([
        evaluate_eer(single_ref_ckpt[order[j]], test_env=order[j], run=run, cache=cache)
        for j in range(K)
    ])

    return compute_cl_metrics(
        perf_matrix,
        joint_ref    = joint_ref_arr,
        single_ref   = single_ref_arr,
        lower_is_better = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    wandb.login()
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    cache = _load_json(EER_CACHE_PATH)     # persistent EER cache

    # ── Estimate computational scale ──────────────────────────────────────
    num_single   = len(ENVS)                              # 4
    num_joint    = sum(1 for s in range(2, len(ENVS)+1)
                       for _ in combinations(ENVS, s))    # 11
    num_cl_steps = (12 + 24 + 24) * CFG["num_runs"]      # 300  (prefix-cached)
    total_trains = (num_single + num_joint) * CFG["num_runs"] + num_cl_steps
    print("=" * 65)
    print(f"CL Baseline  |  Device {CFG['rdevice']}  |  {CFG['target_fs']} Hz  |  "
          f"n_fft={CFG['n_fft']}")
    print(f"Orders: {len(ALL_ORDERS)}  ×  Runs: {CFG['num_runs']}  =  "
          f"{len(ALL_ORDERS)*CFG['num_runs']} CL experiments")
    print(f"Estimated training calls: {total_trains}  "
          f"({total_trains*CFG['max_epochs']} epochs total)")
    print("=" * 65)

    # ── Storage ───────────────────────────────────────────────────────────
    # raw_results[run][order_str][k][env] = EER
    raw_results: dict   = {}
    # per-order metrics across runs
    per_order_metrics: dict[str, dict[str, list]] = {}

    for order in ALL_ORDERS:
        key = _order_to_str(order)
        per_order_metrics[key] = {m: [] for m in ("AA","AIA","FM","BWT","IM","FWT")}

    # ── Loop over runs ─────────────────────────────────────────────────────
    for run in range(CFG["num_runs"]):
        print(f"\n{'━'*65}")
        print(f"  RUN  {run + 1} / {CFG['num_runs']}"
              f"   (seed = {_run_seed(run)})")
        print(f"{'━'*65}")

        # Phase 1 — single-environment reference models
        print("\n[Phase 1] Single-environment reference models")
        single_ref_ckpt = phase1_single_ref(run=run, cache=cache)

        # Phase 2 — joint reference models
        print("\n[Phase 2] Joint reference models")
        fset_to_ckpt = phase2_joint_ref(
            run=run, single_ref_ckpt=single_ref_ckpt, cache=cache
        )

        # Phase 3 — CL fine-tuning for all orderings
        print("\n[Phase 3] CL fine-tuning (all orderings, prefix-cached)")
        run_results = phase3_cl(
            run=run,
            single_ref_ckpt=single_ref_ckpt,
            fset_to_ckpt=fset_to_ckpt,
            cache=cache,
        )
        raw_results[run] = run_results

        # Compute CL metrics for every ordering
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

    # ── Aggregate and report ───────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("RESULTS  —  mean ± std  (across 5 runs × 24 orderings)")
    print("=" * 65)

    # Collect all values across all orderings
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

    # Global summary (pooled across all orders)
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

    # ── Save results ───────────────────────────────────────────────────────
    output = {
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
