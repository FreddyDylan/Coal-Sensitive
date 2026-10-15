# Field-data cases

Two representative field-data cases selected in the accompanying manuscript, the shared experimental calibration parameters and the endpoint result tables for all 212 CT subvolumes. The complete raw CT volumes, field documents and specimen records are not part of this release; the files here cover the inputs and result tables needed for code reuse and reproduction of the reported values.

## Cases

**Case 1 — ROI 100/F007 (100 mm parent core, BX-2 loading).** Selected because it contains three resolved fracture identifiers (F005, F006, F007), the largest number in the inventory. All three fractures carry the same local response law, so their different contributions isolate their hydraulic positions within one geometry.

**Case 2 — ROI 60/F039 (60 mm subcore, BX-2 loading, x direction).** Selected because it showed the largest change in fracture energy fraction between 3.70 and 20.00 MPa among all ROI, response-class and direction combinations.

## Provenance

- CT sources: nominal 60 mm subcore and 100 mm parent-core datasets; processed voxel spacings 45.265 and 87.035 µm. Each ROI is a 20 x 20 x 20 voxel window (edge lengths about 0.905 and 1.741 mm).
- `case_*/labels.npy`: reviewed model label volumes, byte-identical to the inputs of the internal 212-ROI recomputation. Label conventions: `0` ordinary background, `5`/`6`/`7` (or `39`) resolved physical fracture identifiers, `1000000` unresolved high-gray voxels assigned to the effective background with no fracture ownership.
- `case_*/results_series.csv` and `results_contributions.csv`: exported from the internal per-case result files. `run_real_cases.py` reproduces them with the released code (see Verification below).
- `calibration/calibration_curves.csv`: confirmed first-loading effective-stress/permeability observations of BX-2, BX-4, BX-8 and the split/reclosed unfilled BX-5 control (revised 2026-10-07 experimental record).
- `results_212/`: endpoint (20 MPa) results of the full inventory, as summarized in the manuscript tables.

| Case | Label file SHA-256 |
|---|---|
| `case_100_F007` | `44f13279e6e81fec877d0c06d80806ed9a7ad5cd8018c60877adaa6eb4d122ad` |
| `case_60_F039` | `b691d64d0bd34dc3e532fe96ab37fa543e12e933759af7a1add819d917150401` |

## Files

| File | Contents |
|---|---|
| `calibration/calibration_curves.csv` | First-loading curves: sample, branch, effective stress (MPa), permeability (mD), confining and inlet pressures (MPa) |
| `calibration/calibration_geometry.json` | Specimen geometry, aperture and fracture area fraction `f` per sample; BX-5 effective-background documentation |
| `case_100_F007/labels.npy` | Label volume, int32, 20 x 20 x 20, axes in [z, y, x] order |
| `case_100_F007/case_config.json` | Case geometry, stress path, phase map, calibration constants and provenance hashes |
| `case_100_F007/results_series.csv` | 10 stress states x 3 directions: K, retention, loss, S (left/right), background contribution, fracture energy fraction, N_eff |
| `case_100_F007/results_contributions.csv` | Per state x direction x owner: energy weight and J (left/right); the background row completes each sum |
| `case_60_F039/...` | Same layout for the subcore case |
| `results_212/results_212_directional.csv` | 20 MPa endpoint per ROI (and the 2 controls) per direction, 1914 rows |
| `results_212/results_212_medians.csv` | Per ROI and response class, medians over the three directions, 636 rows |
| `results_212/results_212_summary.csv` | Mean and sample standard deviation by CT source and response class (manuscript summary table), 6 rows |
| `run_real_cases.py` | Reproduction driver for the two cases; `--verify` compares with the shipped tables |

## How to run

Python 3.10 or newer with NumPy and SciPy, as for the package itself.

```bash
cd real_cases
python run_real_cases.py --output results --verify
```

The driver rebuilds the stress-dependent conductances from the released curves, solves the condensed voxel network at every stress state and writes `case_100_F007_series.csv`, `case_100_F007_contributions.csv` and the corresponding `case_60_F039` tables into `--output`. With `--verify`, the computed values are compared against the shipped `results_*.csv`.

## Conventions

- Effective stress is the confirmed experimental axis, sigma = P_confining - (P_inlet + P_back) / 2, first loading only.
- Stress path: 3.70, 5.00, 5.25, 7.00, 7.25, 10.00, 10.25, 15.00, 15.75, 20.00 MPa (reported design targets and recomputed effective stresses).
- Permeability unit conversion: 1 mD = 9.869233e-16 m^2, the constant used by the pipeline.
- Fracture phase prior: the effective filled-fracture response from the separated parallel-component calibration K_specimen(sigma) = (1 - f) K_background(sigma) + f K_fracture(sigma), with `f` from the specimen geometry and nominal aperture. Background phase prior: the BX-5 split/reclosed control response as the effective background proxy. Both responses are rebased to the common reference stress (3.70 MPa) before solving, and phase baseline permeabilities use the curve values at that reference.
- Directional sensitivity S_d = -d(ln K_d)/d(sigma) in MPa^-1. Contributions J are additive: fracture contributions plus the background term sum to S_d at each state; energy weights give each owner's share of the directional dissipation.
- The solved quantities are model predictions under the stated calibration and affine boundary conditions; scope and limitations are discussed in the manuscript.

## Verification

The released run reproduces the shipped tables with a worst relative deviation of 1.1e-13 (100/F007 series), 1.5e-11 (100/F007 contributions, dominated by the F007 terms of order 1e-7 MPa^-1) and 4.8e-14 (60/F039 series and contributions).

Endpoint values at 20 MPa, as reproduced from the released inputs:

- 100/F007 (BX-2): z direction K = 4.002034 mD, loss 95.089 %, S = 0.41541412 MPa^-1, fracture energy fraction 0.9793. The additive decomposition along z gives J(F005) = 0.366355, J(F006) = 0.046541, J(F007) = 6.42e-7 and background 0.002518 MPa^-1, matching the manuscript.
- 60/F039 (BX-2): x direction K = 0.158798 mD (from 1.966234 mD at 3.70 MPa), loss 91.924 %, S = 0.24582912 MPa^-1; the fracture energy fraction falls from 0.690676 to 0.414461 over 3.70--20.00 MPa, the largest change in the inventory.

## License

The field-data case files are released by the authors for research reuse and are covered by the repository [LICENSE](../LICENSE) (MIT).
