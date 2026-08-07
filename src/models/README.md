# Model code layout

The maintained global-model code uses one shallow package level:

- `data_loader.py`, `evaluation.py`: shared data and metric contracts.
- `training/`: global baselines, MLP training, ablation and ONNX export.
- `deployment/`: C++/Fortran wrappers and Windows validation scripts.
- `legacy/`: archived exploratory MLP/RNN scripts.

Run entrypoints from the repository root as modules:

```bash
conda run -n buoy-drifter python -m src.models.training.run_ablation --help
```

Tests remain in the top-level `tests/` directory.
