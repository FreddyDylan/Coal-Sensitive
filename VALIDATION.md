# Verification record

Date: 2026-10-15. Software version: 0.1.0.

## Verified environment

- Platform: macOS, Apple Silicon (arm64).
- Python: 3.12.14.
- NumPy: 2.2.6; SciPy: 1.17.1.
- Numerical libraries: the OpenBLAS wheels pinned by SHA-256 in `requirements-macos-py312.txt`.

A fresh virtual environment was created, the software was built into a wheel and installed, and the tests and examples were copied into a separate temporary directory containing no source module. The installed package passed **24 automated tests** and **five synthetic example cases**. The installed `coal-sensitive` command also produced the five expected loading states.

```bash
python -m pip install -r requirements-macos-py312.txt .
python -W error::RuntimeWarning -m unittest -v test_coal_sensitive
python -W error::RuntimeWarning run_examples.py
coal-sensitive example_case.json --output results/series.json
```

For the heterogeneous voxel example over 3.70, 10.00 and 20.00 MPa:

- Maximum relative flow-conservation residual: 1.012e-14.
- Maximum relative mismatch between internal dissipation and boundary work: 1.184e-15.

The explicit graph and condensed voxel solver agree on permeability tensors, their stress derivatives and individual fracture contributions. Finite differences independently check the derivatives. Series, parallel and homogeneous-volume solutions are checked against closed-form expressions. Assertions and their tolerances are available in `test_coal_sensitive.py`.

## Implementation provenance

The `Graph`, `Solution`, `solve`, conductance functions and labelled-volume adapter retain the numerical implementation of the research code. The `CondensedVoxel` constructor and solve kernel retain its exact face elimination and original half-channel contribution recovery. Their Python abstract syntax trees were compared during packaging and matched. The public first-loading response and JSON interfaces expose these kernels without dependencies on project directories or private datasets.

## Platform note

An initial Python 3.10 installation on this Apple Silicon machine selected a SciPy 1.15.3 wheel that failed to load. A SciPy 1.14.1 attempt loaded but the accompanying numerical-library combination raised matrix-multiplication warnings during strict tests. The Python 3.12 OpenBLAS environment above passed with runtime warnings treated as errors. Use that recorded environment to reproduce this local verification.

## Interpretation

These results verify numerical implementation with synthetic inputs. Experimental prediction validation requires the measured calibration curves, reviewed CT labels and the corresponding experimental comparison protocol described in the research study.
