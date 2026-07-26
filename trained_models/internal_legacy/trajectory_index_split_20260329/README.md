# Internal Legacy Model: Trajectory-Index Split

Status: **internal only / superseded**

This directory contains the model released on 2026-03-29 using a split based on
subtrajectory list indices. Later auditing found that multiple subtrajectories
from the same physical buoy `original_ID` could enter different train,
validation, and test sets.

The files are retained only for reproducibility of the first Windows
Python-ONNX-C++-Fortran integration test. They must not be used for:

- final paper metrics or figures;
- new scientific conclusions;
- a new Windows deployment;
- combination with the paper-final scaler or checkpoint.

Archived metrics as originally reported:

- MLP validation joint R2: 0.1428
- MLP test joint R2: 0.1253
- MLP test RMSE: 0.2014 m/s
- Linear test joint R2: 0.1229
- Linear test RMSE: 0.2017 m/s

The checkpoint, scaler, and ONNX in this directory are a matched legacy set.
