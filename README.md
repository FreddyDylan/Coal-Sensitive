# Coal-Sensitive

**A hydraulic network model for coal permeability, stress sensitivity and fracture contributions.**

Coal-Sensitive connects local stress responses to directional permeability through a solved flow field. It represents resolved fractures and effective background pathways in one hydraulic network, updates their conductances with stress, and quantifies each fracture's contribution to the overall sensitivity.

The repository provides a compact Python reference implementation, analytical benchmarks, synthetic three-dimensional examples and two representative field-data cases. The graph solver and the exact voxel face-elimination kernel are carried over from the REV55 research implementation.

## What the software computes

- A symmetric, three-dimensional permeability tensor under affine pressure boundary conditions.
- Directional permeability and its stress derivative.
- Flow-dependent sensitivity contributions for individual fractures and background pathways.
- Effective fracture responses identified from bulk specimen and background curves under a specified parallel-component geometry.
- Equivalent solutions using an explicit hydraulic graph or a voxel solver with internal face nodes eliminated exactly.

## Quick start

Python 3.10 or newer is required; Python 3.12 is recommended. NumPy and SciPy are the runtime dependencies; no GPU is needed.

```bash
git clone https://github.com/FreddyDylan/Coal-Sensitive.git
cd Coal-Sensitive
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m unittest -v test_coal_sensitive
python run_examples.py
```

For Apple Silicon, create the environment with `python3.12 -m venv .venv` and use the tested OpenBLAS builds. Replace the installation command above with:

```bash
python -m pip install -r requirements-macos-py312.txt .
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell. Use `python3` in place of `python` if required by your installation.

Run the standalone JSON case with either command:

```bash
python -m coal_sensitive example_case.json --output results/series.json
coal-sensitive example_case.json --output results/series.json
```

The example suite writes `results/verification.json`. It includes check outcomes and detailed permeability, derivative, contribution and conservation results. Typical execution of these small cases takes a few seconds on a laptop.

The two field-data cases under [`real_cases/`](real_cases/README.md) are reproduced with `python real_cases/run_real_cases.py --output results --verify`.

## Model

### Local response and experimental constraints

On a fixed graph, the conductance of edge $e$ is updated as

$$g_e(\sigma)=g_e(\sigma_0)\frac{k_{a(e)}(\sigma)}{k_{a(e)}(\sigma_0)},$$

where $a(e)$ identifies its response class. `LogLinearResponse` interpolates positive permeability observations in log space and returns the local coefficient $c_e=-\partial\ln g_e/\partial\sigma$.

For a specimen with a known fracture area fraction $f$ and parallel flow through the fracture and background, `SeparatedResponse` evaluates

$$k_f(\sigma)=\frac{k_{\mathrm{bulk}}(\sigma)-(1-f)k_b(\sigma)}{f}.$$

The observed bulk and effective-background curves are interpolated before subtraction. Their derivatives provide the derivative of the separated response. The background curve represents the effective control response used in the calibration. Its physical interpretation depends on the control specimen and its preparation.

### Network solution and contribution analysis

Let $B$ be the node–edge incidence matrix, $G=\mathrm{diag}(g_e)$, and $Y$ the three harmonic-coordinate solutions. The solver imposes the spatial coordinates on the exterior boundary and solves flow conservation at interior nodes. With $Z=B^T Y$, viscosity $\mu$, and sample volume $V$,

$$\mathbf K=\frac{\mu}{V}Z^T G Z, \qquad
\frac{\partial\mathbf K}{\partial\sigma}=-\frac{\mu}{V}Z^T\mathrm{diag}(g_e c_e)Z.$$

For a unit direction $\mathbf d$, define $K_d=\mathbf d^T\mathbf K\mathbf d$. The directional log sensitivity is

$$S_d=-\frac{\partial\ln K_d}{\partial\sigma}=\sum_e w_{e,d}c_e, \qquad
w_{e,d}=\frac{g_e(Z_e\mathbf d)^2}{\sum_j g_j(Z_j\mathbf d)^2}.$$

The weights are recomputed from the current flow field. Summing $w_{e,d}c_e$ over the edges assigned to one fracture gives that fracture's additive contribution. Background channels complete the sum. A direction with zero permeability has an explicit status and an undefined sensitivity represented by JSON `null`.

### Voxel representation

`graph_from_labels` builds a finite-volume graph from an integer array in **[z, y, x]** order. Each cell owns its half-channel to a shared face, retaining its response class and fracture identity. `CondensedVoxel` eliminates the internal degree-two face nodes and reconstructs the energy of the original half-channels for contribution analysis.

The explicit adapter supports exterior voxels labelled `-1` and voxel spacing in **[x, y, z]** order. It applies affine pressure on every exposed voxel face, including any exposed cavity surface. Its default limit is 20,000 active cells. The condensed solver uses fully occupied rectangular arrays, isotropic voxel spacing and strictly positive phase permeability. Both impose affine conditions on all exterior faces; this boundary condition differs from a two-face experiment with sealed sidewalls.

## Examples and verification

The five verification cases listed below use **synthetic** numerical inputs, fully specified in this repository.

| Case | Independent check | Purpose |
|---|---|---|
| Two channels in series | Reciprocal resistance sum; resistance-weighted sensitivity | Verify bottleneck control and stress-dependent redistribution |
| Two channels in parallel | Conductance sum; conductance-weighted sensitivity | Verify alternative flow paths |
| Homogeneous voxel volume | $\mathbf K=k\mathbf I$ | Verify geometry, units and boundary treatment |
| Heterogeneous voxel volume | Explicit graph versus exact face elimination; finite differences | Verify tensor derivatives and individual fracture contributions |
| Specimen component separation | Reconstructed bulk permeability and its derivative | Verify how specimen constraints define a local response |

For the series example at 3.70 MPa, $K_x=2.285714285714286\times10^{-12}$ m² and $S_x=0.24285714285714285$ MPa⁻¹. The complete analytical reference is in [`expected_series.json`](expected_series.json); its relative comparison tolerance is $10^{-10}$.

The automated suite also checks conservation, the network energy identity, tensor symmetry and positive semidefiniteness, disconnected components, coordinate transformations, one-sided derivatives at response knots, input validation, contribution closure and the command-line interface. Tests use analytical formulas and finite differences as independent references.

## Field-data cases

[`real_cases/`](real_cases/README.md) contains two representative field-data cases selected in the accompanying manuscript — ROI 100/F007 (three resolved fractures) and ROI 60/F039 under BX-2 loading — together with the shared experimental calibration parameters and the endpoint result tables for all 212 CT subvolumes. [`real_cases/run_real_cases.py`](real_cases/run_real_cases.py) rebuilds each case from the released label volume and calibration curves and reproduces the shipped result tables to a worst relative deviation of $1.5\times10^{-11}$. See [`real_cases/README.md`](real_cases/README.md) for case selection, provenance, units and conventions.

## Using your own data

For an explicit graph, follow [`example_case.json`](example_case.json):

| Field | Meaning / unit |
|---|---|
| `coordinates_m` | Node coordinates, $N\times3$, metres |
| `edges` | Zero-based node pairs, $M\times2$ |
| `boundary` | Boolean mask identifying affine-pressure boundary nodes |
| `g0_m3_Pa_s` | Conductance at `reference_MPa`, m³ Pa⁻¹ s⁻¹ |
| `response_classes` | One response-library key per edge |
| `fracture_ids` | Physical fracture identifier per edge; `null` for background |
| `volume_m3` | Represented sample volume, m³ |
| `viscosity_Pa_s` | Dynamic viscosity, Pa s |
| `responses` | Stress knots in MPa and permeability values in one consistent unit |
| `stress_path_MPa` | A nondecreasing first-loading path starting at `reference_MPa` |
| `directions` | Named nonzero direction vectors; normalized internally |

The CLI uses the right derivative at internal response knots and the inward derivative at the upper endpoint. The Python API also accepts `side="left"`.

For a labelled volume:

```python
import numpy as np
from coal_sensitive import PhasePrior, graph_from_labels, solve

labels = np.zeros((4, 5, 6), dtype=int)
labels[:, 2, :] = 1
phases = {
    0: PhasePrior("background", 2e-14, None, "synthetic background"),
    1: PhasePrior("fracture", 8e-13, "F1", "synthetic fracture plane"),
}
graph = graph_from_labels(labels, [1e-3, 1e-3, 1e-3], phases, 1e-3)
solution = solve(graph)
print(solution.permeability_m2)
```

Supply reviewed segmentation labels, hydraulic priors, response-class assignments and physical fracture identities for a CT-derived application. A response class can be shared by several fractures; the fracture identifiers determine how their contributions are grouped. The condensed solver's phase priors must refer to the first stress knot of every assigned response. Rebase curves to a common reference before combining curves with different starting stresses.

## Scientific scope and data availability

This release implements linear, steady, single-phase flow on fixed geometry during first loading. Stress enters through local conductance functions. Hydraulic priors and the effective-background calibration determine the physical interpretation of the prediction. The representative volume scale must be assessed for each dataset.

The bundled cases verify the numerical method and illustrate how local responses propagate to the sample scale. Two representative field-data cases, the experimental calibration parameters and the endpoint result tables supporting the manuscript statistics are included under `real_cases/`; the complete research CT volumes, field documents and specimen records remain outside this release. Unloading, adsorption-induced deformation, evolving topology and multiphase transport require additional constitutive models.

## Repository contents

| File | Contents |
|---|---|
| `coal_sensitive.py` | Network and voxel solvers, response calibration, JSON interface |
| `test_coal_sensitive.py` | Automated numerical and input-validation tests |
| `run_examples.py` | Five runnable synthetic verification cases |
| `example_case.json` | Explicit series network and first-loading response library |
| `expected_series.json` | Analytical expected values |
| `real_cases/` | Two field-data cases, shared calibration parameters and 212-subvolume result tables |
| `pyproject.toml`, `requirements.txt` | Installation metadata and dependencies |
| `requirements-macos-py312.txt` | Tested Apple Silicon wheels for Python 3.12 |
| `VALIDATION.md` | Verification commands and tested environments |
| `CITATION.cff` | Software citation metadata |
| `LICENSE` | MIT license |

## Research context and citation

The software accompanies the research manuscript *An experimentally constrained network model for predicting the stress sensitivity of coal permeability and quantifying fracture contributions*. Its authors are Yu Fang, Minjun Chen, Xueni Chen, Yan Chen and Xin Wang.

When using this release, cite the software with its version and commit identifier. GitHub's **Cite this repository** menu reads [`CITATION.cff`](CITATION.cff). Manuscript publication metadata will be added when available.

Research contact: **Yu Fang**, Southwest Petroleum University — [fangyu@swpu.edu.cn](mailto:fangyu@swpu.edu.cn). For software questions and reproducible issues, use [GitHub Issues](https://github.com/FreddyDylan/Coal-Sensitive/issues).

## License

The source code, documentation and bundled synthetic examples are distributed under the [MIT License](LICENSE).
