# Domain-incremental Multi-Channel Replay Speech Detection

Continual learning benchmark for multi-channel audio anti-spoofing on the **ReMASC** dataset. Evaluates four methods on domain-incremental learning across four acoustic environments.

## Authors
Michael Neri, Riccardo Casciotti

_Faculty of Information Technology and Communication Sciences, Tampere University, Tampere, Finland_



## Setup

```bash
conda env create -f environment.yml
conda activate deeplearning
```

## Dataset

Place `Remasc_baseline_backup.zip` in the project root. The dataset is loaded directly from the zip — no extraction needed.

All experiments use **Device 2** (RES_4, 4 channels, 44.1 kHz → resampled to 16 kHz).

## Experiments

### Naive fine-tuning baseline

```bash
python train_cl.py
```

Results saved to `cl_results.json`.

### CL algorithms (EWC / GPM / TSB)

Set `ALGORITHM` at the top of `training_cl_algorithms.py` to `"ewc"`, `"gpm"`, or `"tsb"`, then:

```bash
python training_cl_algorithms.py
```

Results saved to `cl_results_{algorithm}.json`.

Both scripts support **resume**: if a run is interrupted, re-running picks up from the last completed checkpoint automatically.

## Protocol

- **24 orderings** (4!) × **5 runs** = 120 CL experiments per algorithm. It takes approximately 1 week on a NVIDIA GeFORCE RTX 4070.
- **Prefix caching**: models trained on shared prefixes are reused across orderings
- **EER cache**: test evaluations are cached in `eer_cache.json` to avoid re-evaluation

## Evaluation

```bash
# Generate LaTeX tables (Tables 1 and 2 in the paper)
python make_cl_tables.py

# Generate ordering analysis figure
python plot_ordering_analysis.py
```

Outputs: `cl_paper_tables.tex`, `ordering_analysis.pdf`

## CL Metrics

Computed by `cl_metrics.py` from the 4×4 performance matrix `perf[k, j]` = EER on environment `j` after training on environments `0..k`.

| Metric | Meaning | Direction |
|--------|---------|-----------|
| AA | Average EER at final step | lower = better |
| AIA | Average incremental EER across all steps | lower = better |
| FM | Forgetting measure | lower = better |
| BWT | Backward transfer | higher = better |
| IM | Intransigence vs. joint training | lower = better |
| FWT | Forward transfer vs. single-task | higher = better |

## Files

| File | Role |
|------|------|
| `Remasc.py` | Dataset class and data transforms |
| `model.py` | Baseline model (beamformer + CNN + GRU) |
| `model_cl.py` | CL modules: `EWCModelModule`, `GPMModelModule`, `TSBModelModule` |
| `train_cl.py` | Naive fine-tuning baseline |
| `training_cl_algorithms.py` | CL algorithm training |
| `cl_metrics.py` | CL evaluation metrics |
| `utils.py` | EER metric |
| `make_cl_tables.py` | LaTeX table generator |
| `plot_ordering_analysis.py` | Ordering analysis figure |

--------

If you use any part of this code, please cite the following works regarding the spatial deepfake detector and the domain-incremental learning

```
@ARTICLE{Neri_spatial_deepfake_2025,
  author={Neri, M. and Virtanen, T.},
  journal={IEEE Open Journal of Signal Processing}, 
  title={Multi-Channel Replay Speech Detection Using an Adaptive Learnable Beamformer}, 
  year={2025},
  volume={6},
  number={},
  pages={530-535},
  doi={10.1109/OJSP.2025.3568758}}

@INPROCEEDINGS{Neri_CL_2026,
  title={Domain-Incremental Learning for Multi-channel Replay Speech Detection},
  booktitle={}, 
  author={Michael Neri and Riccardo Casciotti},
  year={2026},
  volume={},
  number={},
  pages={},
  doi={}}

```

