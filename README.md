# Time-Series-MiniRocket

Reproduction and extension of results from:

> Dempster, A., Schmidt, D.F., Webb, G.I. (2021). *MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification*. KDD 2021. [arXiv:2012.08791](https://arxiv.org/abs/2012.08791) &middot; [original code](https://github.com/angus924/minirocket)

## Layout

- `code/` — MiniRocket implementation (`minirocket.py`, `minirocket_multivariate.py`, `minirocket_variable.py`, `minirocket_dv.py`, `softmax.py`), copied from the authors' repository, plus our own improvement work in `code/results_improvements/` (`minirocket_aa.py`, `minirocket_poc.py` and the notebooks that test them against all UCR109 resamples).
- `results/` — the authors' originally published results (`accuracy_ucr109_resamples.csv`, `results_ucr109_mean.csv`, `time_training_*.csv`), used as the baseline we reproduce against.
- `notebooks/` — our reproduction notebooks:
  - `MiniRocket_UCR109_Benchmark.ipynb` — accuracy/timing on all 109 UCR archive datasets.
  - `MiniRocket_Scalability_Reproduction.ipynb` — training-set-size scaling (InsectSound, FruitFlies, MosquitoSound) and series-length scaling (DucksAndGeese).
  - `MiniRocket_Ablation.ipynb` — ablation study over MiniRocket's components.
  - `MiniRocket_Robustness.ipynb` — robustness checks (e.g. seed/noise sensitivity).
- `reproduction_results/` — outputs of the notebooks above: per-dataset/per-resample CSVs, summary tables, plots, experiment metadata, comparisons against the authors' published numbers, train/test split indices (`ucr109_results/splits/`), and the prepared data artifacts needed to re-run the scalability experiments on identical splits (`scalability_prepared/`: shuffle order, class-label maps, dataset-prep metadata).

## Datasets

The raw datasets are **not** committed to this repository (too large). They are the same datasets used in the original MiniRocket paper:

- **UCR Time Series Classification Archive (2018)** — all 109/128 univariate datasets used for the UCR109 benchmark.
  - Official archive: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
  - Mirror with per-dataset download: https://www.timeseriesclassification.com/dataset.php
  - Citation: Dau, H.A. et al. (2019). *The UCR Time Series Archive*. IEEE/CAA Journal of Automatica Sinica.
- **Scalability-experiment datasets** (used for the training-set-size and series-length experiments), from the same archive family:
  - [InsectSound](https://www.timeseriesclassification.com/description.php?Dataset=InsectSound) — 50,000 instances, 10 classes, length 600.
  - [FruitFlies](https://www.timeseriesclassification.com/description.php?Dataset=FruitFlies) — 34,518 instances, 3 classes, length 5,000.
  - [MosquitoSound](https://www.timeseriesclassification.com/description.php?Dataset=MosquitoSound) — 279,566 instances, 6 classes, length 3,750.
  - [DucksAndGeese](https://www.timeseriesclassification.com/description.php?Dataset=DucksAndGeese) — 100 instances, 5 classes, length 236,784.

To re-run the notebooks, download the datasets from the links above into a local `data/` folder (not tracked by git) and point each notebook's data-path cell at it.
