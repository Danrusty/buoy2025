# Model code layout

The model code uses one shallow package level:

- `data_loader.py`, `evaluation.py`: shared data and metric contracts.
- `training/`: global baselines, MLP training, ablation and ONNX export.
- `regional/`: row-level regional datasets and regional MLP studies.
- `adapters/`: frozen-global low-order adapter studies.
- `deployment/`: C++/Fortran wrappers and Windows validation scripts.
- `legacy/`: archived exploratory MLP/RNN scripts.

Run maintained entrypoints from the repository root as modules, for example:

```bash
conda run -n buoy-drifter python -m src.models.training.run_ablation --help
conda run -n buoy-drifter python -m src.models.adapters.run_cms_global_adapter --help
```

Tests remain in the top-level `tests/` directory.
