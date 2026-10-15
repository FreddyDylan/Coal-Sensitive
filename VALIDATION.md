# Verification record

Date: 2026-10-16. Software version: 0.2.0. The synthetic verification below was performed on 2026-10-15 for version 0.1.0.

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

## Field-data cases

The two released field-data cases were rerun with the released code on 2026-10-16 (NumPy 2.2.6, SciPy 1.15.3, Linux x86-64):

```bash
cd real_cases
python run_real_cases.py --output results --verify
```

- 100/F007 (BX-2): worst relative deviation from the shipped tables, 1.1e-13 (series) and 1.5e-11 (contributions; the largest F007 contribution terms are of order 1e-7 MPa^-1).
- 60/F039 (BX-2): worst relative deviation 4.8e-14 (series and contributions).

Endpoint values at 20 MPa reproduce the manuscript: 100/F007 $S_z=0.41541412$ MPa$^{-1}$ with $J_{F005}=0.366355$, $J_{F006}=0.046541$, $J_{F007}=6.42\times10^{-7}$ and background $0.002518$ MPa$^{-1}$; the 60/F039 x-direction fracture energy fraction falls from 0.690676 to 0.414461 between 3.70 and 20.00 MPa. These checks reproduce the manuscript result tables from the released inputs; they are a reproducibility record, not an independent validation of the model on new data.

## Interpretation

These results verify the numerical implementation with synthetic inputs and reproduce the manuscript result tables from the released field-data case inputs. Experimental prediction validation for other datasets requires the corresponding measured calibration curves, reviewed CT labels and the comparison protocol described in the research study.
