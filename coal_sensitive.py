"""Coal permeability, stress sensitivity and fracture contributions.

Lengths, permeability, viscosity and volume use SI units; stress uses MPa.
The graph and voxel kernels originate from the REV55 research implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu, cg

__version__ = "0.2.0"

class ModelInputError(ValueError):
    """An input violates the model's geometry or response requirements."""

@dataclass(frozen=True)
class Graph:
    coordinates_m: np.ndarray
    edges: np.ndarray
    boundary: np.ndarray
    g0_m3_Pa_s: np.ndarray
    response_classes: tuple[str, ...]
    fracture_ids: tuple[str | None, ...]
    volume_m3: float
    viscosity_Pa_s: float
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        raw_edges = np.asarray(self.edges)
        if raw_edges.size and (not np.issubdtype(raw_edges.dtype, np.number) or
                               not np.all(np.isfinite(raw_edges)) or np.any(raw_edges != np.floor(raw_edges))):
            raise ModelInputError("Edge endpoints must be integer node indices")
        raw_boundary = np.asarray(self.boundary)
        if not np.all(np.isin(raw_boundary, [False, True])):
            raise ModelInputError("Boundary must be a boolean mask")
        for key, dtype in (("coordinates_m", float), ("edges", int), ("boundary", bool), ("g0_m3_Pa_s", float)):
            a = np.array(getattr(self, key), dtype=dtype, copy=True)
            if key == "edges" and a.size == 0:
                a = a.reshape(0, 2)
            a.setflags(write=False)
            object.__setattr__(self, key, a)
        x, e, b, g = self.coordinates_m, self.edges, self.boundary, self.g0_m3_Pa_s
        if x.ndim != 2 or x.shape[1] != 3 or len(x) < 2 or not np.all(np.isfinite(x)):
            raise ModelInputError("Coordinates must be a finite N x 3 array, N >= 2")
        if e.ndim != 2 or e.shape[1] != 2 or np.any(e < 0) or np.any(e >= len(x)):
            raise ModelInputError("Edges must be M x 2 valid node indices")
        if len(e) and (np.any(e[:, 0] == e[:, 1]) or np.any(np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1) == 0)):
            raise ModelInputError("Zero-length or self-loop hydraulic edges are not allowed")
        if b.shape != (len(x),) or not np.any(b):
            raise ModelInputError("A boundary mask with at least one anchored node is required")
        if g.shape != (len(e),) or not np.all(np.isfinite(g)) or np.any(g < 0):
            raise ModelInputError("Baseline edge conductance must be finite and nonnegative")
        for name in ("volume_m3", "viscosity_Pa_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ModelInputError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        for name in ("response_classes", "fracture_ids"):
            value = tuple(getattr(self, name))
            if len(value) != len(e):
                raise ModelInputError(f"{name} must have one item per edge")
            object.__setattr__(self, name, value)
        if any(not isinstance(c, str) or not c for c in self.response_classes):
            raise ModelInputError("Each edge needs an explicit response class")
        if any(f is not None and (not isinstance(f, str) or not f) for f in self.fracture_ids):
            raise ModelInputError("Fracture ID must be a nonempty string or null for nonfracture edges")


def unit_direction(direction) -> np.ndarray:
    d = np.asarray(direction, dtype=float)
    if d.shape != (3,) or not np.all(np.isfinite(d)) or np.linalg.norm(d) == 0:
        raise ModelInputError("Direction must be a finite, nonzero 3-vector")
    return d / np.linalg.norm(d)


@dataclass
class Solution:
    graph: Graph
    conductance: np.ndarray
    harmonic_coordinates: np.ndarray
    edge_coordinate_drop: np.ndarray
    permeability_m2: np.ndarray
    diagnostics: dict

    def derivative(self, coefficients: np.ndarray) -> np.ndarray:
        c = np.asarray(coefficients, dtype=float)
        if c.shape != self.conductance.shape or not np.all(np.isfinite(c)):
            raise ModelInputError("One finite sensitivity coefficient per edge is required")
        z = self.edge_coordinate_drop
        return -self.graph.viscosity_Pa_s / self.graph.volume_m3 * (z.T * (self.conductance * c)) @ z

    def directional(self, direction, coefficients: np.ndarray) -> dict:
        d = unit_direction(direction)
        c = np.asarray(coefficients, float)
        self.derivative(c)  # shared shape/finite validation
        dz = self.edge_coordinate_drop @ d
        energy = self.conductance * dz ** 2
        total = float(energy.sum())
        geometry_scale = max(float(np.ptp(self.graph.coordinates_m, axis=0).max()), np.finfo(float).tiny)
        zero = total == 0 or np.max(np.abs(dz[self.conductance > 0]), initial=0) <= 128 * np.finfo(float).eps * geometry_scale
        if zero:
            return {"status": "zero_directional_permeability", "K_m2": 0.0,
                    "S_per_MPa": None, "edge_weights": None, "fracture_contributions": {},
                    "background_contribution_per_MPa": None, "N_eff_fractures": None,
                    "fracture_energy_fraction": None}
        w = energy / total
        contributions = {}
        for owner in sorted({f for f in self.graph.fracture_ids if f is not None}):
            mask = np.array([f == owner for f in self.graph.fracture_ids])
            contributions[owner] = {"energy_weight": float(w[mask].sum()),
                                    "J_per_MPa": float(w[mask] @ c[mask])}
        fracture_fraction = sum(v["energy_weight"] for v in contributions.values())
        n_eff = (fracture_fraction ** 2 / sum(v["energy_weight"] ** 2 for v in contributions.values())
                 if fracture_fraction > 0 else None)
        background = np.array([f is None for f in self.graph.fracture_ids])
        return {"status": "ok", "K_m2": self.graph.viscosity_Pa_s / self.graph.volume_m3 * total,
                "S_per_MPa": float(w @ c), "edge_weights": w.tolist(),
                "fracture_contributions": contributions,
                "background_contribution_per_MPa": float(w[background] @ c[background]),
                "N_eff_fractures": n_eff, "fracture_energy_fraction": fracture_fraction}


def solve(graph: Graph, conductance=None) -> Solution:
    """One sparse numerical LU factorization, three harmonic-coordinate RHS.

    Components without Dirichlet boundary have a constant gauge and zero flow.
    Baseline zero edges stay available in the graph but are not used to anchor
    components. No artificial diagonal leakage or dense Schur inverse is added.
    """
    g = np.array(graph.g0_m3_Pa_s if conductance is None else conductance, dtype=float, copy=True)
    if g.shape != graph.g0_m3_Pa_s.shape or np.any(g < 0) or not np.all(np.isfinite(g)):
        raise ModelInputError("Conductance must be finite, nonnegative, and match the graph")
    n, m = len(graph.coordinates_m), len(graph.edges)
    active = g > 0
    e = graph.edges
    B = sparse.coo_matrix((np.r_[np.ones(m), -np.ones(m)],
                           (np.r_[e[:, 0], e[:, 1]], np.tile(np.arange(m), 2))), shape=(n, m)).tocsr()
    ae = e[active]
    adjacency = sparse.coo_matrix((np.ones(2 * len(ae)),
                                   (np.r_[ae[:, 0], ae[:, 1]], np.r_[ae[:, 1], ae[:, 0]])), shape=(n, n)).tocsr()
    count, labels = connected_components(adjacency, directed=False)
    anchored_labels = np.unique(labels[graph.boundary])
    anchored = np.isin(labels, anchored_labels)
    interior = np.flatnonzero(anchored & ~graph.boundary)
    boundary = np.flatnonzero(graph.boundary)
    # Remove an arbitrary coordinate offset to reduce cancellation under translation.
    origin = graph.coordinates_m[boundary].mean(axis=0)
    Y = np.zeros((n, 3))
    Y[boundary] = graph.coordinates_m[boundary] - origin
    scale = float(g.max(initial=0)) or 1.0
    L = (B @ sparse.diags(g / scale) @ B.T).tocsr()
    factorization_count = 0
    if len(interior):
        lu = splu(L[interior][:, interior].tocsc())
        Y[interior] = lu.solve(-L[interior][:, boundary] @ Y[boundary])
        factorization_count = 1
    if not np.all(np.isfinite(Y)):
        raise ArithmeticError("Nonfinite harmonic-coordinate solution")
    Z = B.T @ Y
    K = graph.viscosity_Pa_s / graph.volume_m3 * (Z.T * g) @ Z
    K = (K + K.T) / 2
    flow = g[:, None] * Z
    balance = B @ flow
    residual = float(np.max(np.abs(balance[~graph.boundary]), initial=0))
    flow_scale = float(np.max(np.abs(flow), initial=0))
    relative_residual = residual / flow_scale if flow_scale else 0.0
    energy = Z.T @ flow
    boundary_work = Y[boundary].T @ balance[boundary]
    norm = float(np.linalg.norm(energy))
    energy_error = float(np.linalg.norm(energy - boundary_work) / norm) if norm else 0.0
    eig = np.linalg.eigvalsh(K)
    diagnostics = {"nodes": n, "edges": m, "active_edges": int(active.sum()),
                   "components": int(count), "floating_components": int(count - len(anchored_labels)),
                   "factorizations": factorization_count, "right_hand_sides": 3 if len(interior) else 0,
                   "conservation_relative_residual": relative_residual,
                   "energy_relative_error": energy_error,
                   "K_eigenvalues_m2": eig.tolist(),
                   "positive_semidefinite": bool(eig.min() >= -1e-12 * max(float(np.linalg.norm(K)), np.finfo(float).tiny))}
    if relative_residual > 1e-8 or energy_error > 1e-8 or not diagnostics["positive_semidefinite"]:
        raise ArithmeticError(f"Numerical verification failed: {diagnostics}")
    return Solution(graph, g, Y, Z, K, diagnostics)


def porous_conductance(permeability_m2, area_m2, length_m, viscosity_Pa_s):
    values = np.array([permeability_m2, area_m2, length_m, viscosity_Pa_s], float)
    if not np.all(np.isfinite(values)) or values[0] < 0 or np.any(values[1:] <= 0):
        raise ModelInputError("Porous channel requires K >= 0 and positive area, length, viscosity")
    return float(values[0] * values[1] / (values[2] * values[3]))


def fracture_conductance(aperture_m, width_m, length_m, viscosity_Pa_s):
    """Smooth parallel-plate prior; mechanical aperture is not automatically hydraulic."""
    values = np.array([aperture_m, width_m, length_m, viscosity_Pa_s], float)
    if not np.all(np.isfinite(values)) or values[0] < 0 or np.any(values[1:] <= 0):
        raise ModelInputError("Fracture channel requires b >= 0 and positive width, length, viscosity")
    return float(values[1] * values[0] ** 3 / (12 * values[3] * values[2]))


@dataclass(frozen=True)
class PhasePrior:
    response_class: str
    permeability_m2: float
    fracture_id: str | None
    provenance: str

    def __post_init__(self):
        if not self.response_class or not self.provenance:
            raise ModelInputError("Every phase needs an explicit response class and provenance/scenario description")
        if not np.isfinite(self.permeability_m2) or self.permeability_m2 < 0:
            raise ModelInputError("Phase baseline permeability must be finite and nonnegative")


def graph_from_labels(labels, spacing_xyz_m, phases: dict[int, PhasePrior], viscosity_Pa_s,
                      origin_xyz_m=(0, 0, 0), *, max_cells=20000) -> Graph:
    """Read [z,y,x] integer labels; -1 is outside, every other label is explicit.

    Boundary values are affine pressures at EXTERIOR CELL FACES. Volume is the
    actual voxel-union volume, not an assumed sphere volume. Cropping a sphere
    on this grid is an approximation and requires a subsequent resolution study.
    This adapter does not segment grayscale CT or infer water/filled-fracture K.
    """
    a = np.asarray(labels)
    if a.ndim != 3 or not np.issubdtype(a.dtype, np.integer) or np.any(a < -1):
        raise ModelInputError("CT labels must be a 3D integer array with outside=-1")
    spacing = np.asarray(spacing_xyz_m, float)
    origin = np.asarray(origin_xyz_m, float)
    if spacing.shape != (3,) or np.any(spacing <= 0) or not np.all(np.isfinite(spacing)):
        raise ModelInputError("Supply three positive voxel sizes in x,y,z order (metres)")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ModelInputError("Origin must contain three finite coordinates")
    count = int(np.count_nonzero(a >= 0))
    if count == 0 or count > max_cells:
        raise ModelInputError(f"Small-scale CT adapter requires 1..{max_cells} active cells; got {count}")
    used = set(map(int, np.unique(a[a >= 0])))
    if not used <= phases.keys():
        raise ModelInputError(f"Missing explicit hydraulic priors for labels {sorted(used - phases.keys())}")
    cells = np.argwhere(a >= 0)
    index = {tuple(cell): i for i, cell in enumerate(cells)}
    coordinates = [(origin + cell[::-1] * spacing).tolist() for cell in cells]
    boundary = [False] * count
    edges, g0, classes, owners = [], [], [], []

    def add_half(cell, face, axis):
        cell_id = index[tuple(cell)]
        phase = phases[int(a[tuple(cell)])]
        area = float(np.prod(np.delete(spacing, axis)))
        g = porous_conductance(phase.permeability_m2, area, spacing[axis] / 2, viscosity_Pa_s)
        edges.append((cell_id, face)); g0.append(g)
        classes.append(phase.response_class); owners.append(phase.fracture_id)

    for cell in cells:
        cell_id = index[tuple(cell)]
        for axis in range(3):
            for sign in (-1, 1):
                neighbor = cell.copy(); neighbor[2 - axis] += sign
                adjacent = tuple(neighbor) in index
                if adjacent and sign == -1:
                    continue  # shared face is emitted once, with two owned half-channels
                position = np.asarray(coordinates[cell_id]).copy()
                position[axis] += sign * spacing[axis] / 2
                face = len(coordinates)
                coordinates.append(position.tolist()); boundary.append(not adjacent)
                add_half(cell, face, axis)
                if adjacent:
                    add_half(neighbor, face, axis)
    return Graph(np.array(coordinates), np.array(edges), np.array(boundary), np.array(g0),
                 tuple(classes), tuple(owners), count * float(np.prod(spacing)), viscosity_Pa_s,
                 {"data_kind": "segmented_volume_with_explicit_priors", "cells": count,
                  "input_shape_zyx": list(a.shape), "spacing_xyz_m": spacing.tolist(),
                  "origin_xyz_m": origin.tolist(), "boundary": "affine pressure at exterior voxel faces",
                  "geometry": "voxel union; discretized geometry, not an exact 55 mm sphere",
                  "phase_priors": {str(k): vars(phases[k]) for k in sorted(used)},
                  "prediction_validated": False})


class CondensedVoxel:
    def __init__(self, labels, spacing_m, phases, viscosity=1e-3, method="cg"):
        labels = np.asarray(labels)
        if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
            raise ValueError("Batch solver requires a fully occupied integer voxel array")
        if not np.isfinite(spacing_m) or spacing_m <= 0 or not np.isfinite(viscosity) or viscosity <= 0:
            raise ValueError("Positive finite isotropic spacing and viscosity are required")
        self.phase_labels, self.cell_phase = np.unique(labels, return_inverse=True)
        self.cell_phase = self.cell_phase.ravel()
        self.phases = [phases[int(label)] for label in self.phase_labels]
        if any(p.permeability_m2 <= 0 for p in self.phases):
            raise ValueError("Strictly positive phase permeability is required for exact face elimination")
        self.k0 = np.array([p.permeability_m2 for p in self.phases])
        self.n = labels.size
        self.h = float(spacing_m)
        self.mu = float(viscosity)
        self.volume = self.n * self.h**3
        grid = np.arange(self.n).reshape(labels.shape)
        coords = np.moveaxis(np.indices(labels.shape), 0, -1).reshape(-1, 3)[:, ::-1]
        coords = (coords - (np.array(labels.shape)[::-1]-1)/2) * self.h
        self.method = method
        self.previous_y = coords.copy()
        pairs, cells, values = [], [], []
        for dim in range(3):
            lo, hi = [slice(None)]*3, [slice(None)]*3
            lo[dim], hi[dim] = slice(0, -1), slice(1, None)
            pairs.append(np.column_stack((grid[tuple(lo)].ravel(), grid[tuple(hi)].ravel())))
            for side in (0, -1):
                face = [slice(None)]*3; face[dim] = side
                ids = grid[tuple(face)].ravel()
                y = coords[ids].copy(); y[:, 2-dim] += (-.5 if side == 0 else .5)*self.h
                cells.append(ids); values.append(y)
        self.pairs = np.concatenate(pairs)
        self.i, self.j = self.pairs.T
        self.ext = np.concatenate(cells)
        self.yb = np.concatenate(values)
        self.pi, self.pj, self.pb = self.cell_phase[self.i], self.cell_phase[self.j], self.cell_phase[self.ext]
        self.classes = np.array([p.response_class for p in self.phases])
        self.owners = [p.fracture_id for p in self.phases]
        self.rows = np.r_[self.i, self.j, np.arange(self.n)]
        self.cols = np.r_[self.j, self.i, np.arange(self.n)]

    def solve(self, responses, stress, branch):
        response = {name: (responses[name].evaluate(stress, branch, "left"),
                           responses[name].evaluate(stress, branch, "right")) for name in set(self.classes)}
        ratios = np.array([response[name][0][0] for name in self.classes])
        left = np.array([response[name][0][1] for name in self.classes])
        right = np.array([response[name][1][1] for name in self.classes])
        return self._solve_phase_fields(ratios, left, right, stress, branch)

    def _solve_phase_fields(self, ratios, left, right, coordinate, branch):
        """Shared positive-conductance kernel; callers label derivative units.

        Internal result keys retain the original stress schema. The water
        adapter translates every derivative/contribution before exporting it.
        """
        ratios, left, right = (np.asarray(x, dtype=float) for x in (ratios, left, right))
        if any(x.shape != self.k0.shape or not np.all(np.isfinite(x)) for x in (ratios, left, right)) or np.any(ratios <= 0):
            raise ValueError('One positive response and two finite log derivatives per phase required')
        # Two material-owned half-channel conductances in series.
        half = 2*self.h/self.mu * self.k0 * ratios
        gi, gj, gb = half[self.pi], half[self.pj], half[self.pb]
        g = gi*gj/(gi+gj)
        diagonal = (np.bincount(self.i, weights=g, minlength=self.n)
                    + np.bincount(self.j, weights=g, minlength=self.n)
                    + np.bincount(self.ext, weights=gb, minlength=self.n))
        scale = float(diagonal.max())
        laplacian = sparse.coo_matrix((np.r_[-g, -g, diagonal]/scale, (self.rows, self.cols)),
                                      shape=(self.n, self.n)).tocsc()
        rhs = np.column_stack([np.bincount(self.ext, weights=gb*self.yb[:, d], minlength=self.n)
                               for d in range(3)]) / scale
        iterations = [0, 0, 0]
        fallback = False
        if self.method == "cg":
            preconditioner = sparse.diags(scale/diagonal)
            csr = laplacian.tocsr()
            y = np.empty_like(rhs)
            for d in range(3):
                def count_iteration(_):
                    iterations[d] += 1
                y[:, d], info = cg(csr, rhs[:, d], M=preconditioner, x0=self.previous_y[:, d],
                                  rtol=1e-13, atol=0., maxiter=3000, callback=count_iteration)
                if info != 0:
                    fallback = True
                    break
            if fallback:
                y = splu(laplacian, permc_spec="MMD_AT_PLUS_A").solve(rhs)
        elif self.method == "direct":
            y = splu(laplacian, permc_spec="MMD_AT_PLUS_A").solve(rhs)
        else:
            raise ValueError("Unknown condensed solver method")
        self.previous_y = y.copy()
        if not np.all(np.isfinite(y)):
            raise ArithmeticError("Nonfinite condensed solution")
        drop = y[self.i]-y[self.j]
        ext_drop = y[self.ext]-self.yb
        flux, ext_flux = g[:, None]*drop, gb[:, None]*ext_drop
        balance = np.column_stack([
            np.bincount(self.i, weights=flux[:, d], minlength=self.n)
            - np.bincount(self.j, weights=flux[:, d], minlength=self.n)
            + np.bincount(self.ext, weights=ext_flux[:, d], minlength=self.n) for d in range(3)])
        residual = float(np.abs(balance).max() / max(np.abs(flux).max(initial=0), np.abs(ext_flux).max()))
        # Recover energy of each ORIGINAL material-owned half-channel, not only
        # the equivalent full interface. This preserves per-fracture J exactly.
        ei, ej = g*g/gi, g*g/gj
        tensor_by_phase = np.zeros((len(self.phases), 3, 3))
        for a in range(3):
            for b in range(a, 3):
                cross, ext_cross = drop[:, a]*drop[:, b], ext_drop[:, a]*ext_drop[:, b]
                values = (np.bincount(self.pi, weights=ei*cross, minlength=len(self.phases))
                          + np.bincount(self.pj, weights=ej*cross, minlength=len(self.phases))
                          + np.bincount(self.pb, weights=gb*ext_cross, minlength=len(self.phases)))
                tensor_by_phase[:, a, b] = tensor_by_phase[:, b, a] = values
        energy = tensor_by_phase.sum(axis=0)
        boundary_work = self.yb.T @ (-ext_flux)
        energy_error = float(np.linalg.norm(energy-boundary_work)/np.linalg.norm(energy))
        k = self.mu/self.volume * energy
        phase_k = self.mu/self.volume * tensor_by_phase
        eigen = np.linalg.eigvalsh(k)
        positive = bool(eigen.min() >= -1e-12*np.linalg.norm(k))
        if residual > 1e-8 or energy_error > 1e-8 or not positive:
            raise ArithmeticError(f"Condensed verification failed: residual={residual}, energy={energy_error}")
        derivatives = {"left": -np.einsum("p,pij->ij", left, phase_k),
                       "right": -np.einsum("p,pij->ij", right, phase_k)}
        directions = {}
        for axis, name in enumerate("xyz"):
            weights = phase_k[:, axis, axis] / k[axis, axis]
            d = {"K_m2": float(k[axis, axis]), "status": "ok"}
            fracture_energy = 0.
            for side, coefficients in (("left", left), ("right", right)):
                owner_data = {}
                background = 0.
                for p, owner in enumerate(self.owners):
                    if owner is None:
                        background += float(weights[p]*coefficients[p])
                    else:
                        slot = owner_data.setdefault(owner, {"energy_weight": 0., "J_per_MPa": 0.})
                        slot["energy_weight"] += float(weights[p])
                        slot["J_per_MPa"] += float(weights[p]*coefficients[p])
                sensitivity = float(weights @ coefficients)
                total = background + sum(v["J_per_MPa"] for v in owner_data.values())
                if abs(total-sensitivity) > 1e-12:
                    raise ArithmeticError("Background + physical fracture contribution mismatch")
                d[f"S_{side}_per_MPa"] = sensitivity
                d[f"background_{side}_per_MPa"] = background
                d[f"fracture_contributions_{side}"] = owner_data
                fracture_energy = sum(v["energy_weight"] for v in owner_data.values())
            d["fracture_energy_fraction"] = fracture_energy
            d["N_eff_fractures"] = (fracture_energy**2 / sum(v["energy_weight"]**2 for v in owner_data.values())
                                     if fracture_energy else None)
            directions[name] = d
        return {"stress_MPa": float(coordinate), "branch": branch, "K_m2": k.tolist(),
                "dK_dMPa_left": derivatives["left"].tolist(), "dK_dMPa_right": derivatives["right"].tolist(),
                "directions": directions, "edge_response_envelope": [float(ratios.min()), float(ratios.max())],
                "diagnostics": {"conservation_relative_residual": residual, "energy_relative_error": energy_error,
                                "positive_semidefinite": positive, "K_eigenvalues_m2": eigen.tolist(),
                                "linear_method": self.method, "cg_iterations": iterations, "direct_fallback": fallback,
                                "condensed_unknowns": self.n, "original_nodes": self.n+len(self.i)+len(self.ext),
                                "original_half_edges": 2*len(self.i)+len(self.ext)}}



@dataclass(frozen=True)
class LogLinearResponse:
    """Positive first-loading permeability observations, interpolated in log K.

    All permeability values in a calibration must use the same unit. A response
    ratio is dimensionless; local conductances retain the graph's SI units.
    At an internal knot, ``side`` chooses the one-sided log derivative.
    """
    name: str
    stress_MPa: np.ndarray
    permeability: np.ndarray

    def __post_init__(self):
        for key in ("stress_MPa", "permeability"):
            value = np.array(getattr(self, key), dtype=float, copy=True)
            if value.ndim != 1 or not np.all(np.isfinite(value)):
                raise ModelInputError(f"{self.name}: expected a finite one-dimensional {key}")
            value.setflags(write=False)
            object.__setattr__(self, key, value)
        x, k = self.stress_MPa, self.permeability
        if len(x) < 2 or x.shape != k.shape or np.any(np.diff(x) <= 0):
            raise ModelInputError("Response needs matching arrays with increasing stress")
        if np.any(k <= 0) or np.any(np.diff(k) > 0):
            raise ModelInputError("First-loading permeability must be positive and nonincreasing")

    def at(self, stress_MPa, side="right"):
        """Return absolute permeability and -d(log K)/d(stress), in MPa^-1."""
        s = float(stress_MPa)
        x = self.stress_MPa
        if side not in ("left", "right") or not np.isfinite(s) or not x[0] <= s <= x[-1]:
            raise ModelInputError("Stress must lie within the response interval; side is left or right")
        logs = np.log(self.permeability)
        j = int(np.clip(np.searchsorted(x, s, side=side)-1, 0, len(x)-2))
        return float(np.exp(np.interp(s, x, logs))), float(-(logs[j+1]-logs[j])/(x[j+1]-x[j]))

    def evaluate(self, stress_MPa, branch="L", side="right"):
        if branch != "L":
            raise ModelInputError("This response represents first loading")
        value, coefficient = self.at(stress_MPa, side)
        return value / self.permeability[0], coefficient


class SeparatedResponse:
    """Effective fracture response from a parallel specimen component model.

    K_bulk = (1-f) K_background + f K_fracture, with a known fracture area
    fraction f and an explicit effective-background response. Source curves are
    interpolated BEFORE subtraction; the separated law is evaluated exactly.
    """
    def __init__(self, bulk, background, fraction, name="fracture"):
        if not np.isfinite(fraction) or not 0 < fraction < 1:
            raise ModelInputError("Fracture area fraction must be strictly between zero and one")
        self.bulk, self.background, self.fraction, self.name = bulk, background, float(fraction), name
        low = max(bulk.stress_MPa[0], background.stress_MPa[0])
        high = min(bulk.stress_MPa[-1], background.stress_MPa[-1])
        if low >= high:
            raise ModelInputError("Responses need a common stress interval")
        self.stress_MPa = np.unique(np.r_[low, high, bulk.stress_MPa, background.stress_MPa])
        self.stress_MPa = self.stress_MPa[(self.stress_MPa >= low) & (self.stress_MPa <= high)]
        # Difference of two exponentials on each interval: endpoint checks
        # establish positivity and monotone decay throughout that interval.
        for a, b in zip(self.stress_MPa[:-1], self.stress_MPa[1:]):
            for stress, side in ((a, "right"), (b, "left")):
                if self.at(stress, side)[1] < -1e-12:
                    raise ModelInputError("Separated response increases; review the component assumptions")
        self.permeability = np.array([self.at(s)[0] for s in self.stress_MPa])

    def at(self, stress_MPa, side="right"):
        if not self.stress_MPa[0] <= stress_MPa <= self.stress_MPa[-1]:
            raise ModelInputError("Stress must lie within the shared calibration interval")
        bulk, cb = self.bulk.at(stress_MPa, side)
        background, cg = self.background.at(stress_MPa, side)
        numerator = bulk - (1-self.fraction)*background
        if not np.isfinite(numerator) or numerator <= 0:
            raise ModelInputError("Separated fracture permeability must be positive")
        return numerator/self.fraction, (bulk*cb-(1-self.fraction)*background*cg)/numerator

    def evaluate(self, stress_MPa, branch="L", side="right"):
        if branch != "L":
            raise ModelInputError("Separated response represents first loading")
        value, coefficient = self.at(stress_MPa, side)
        return value / self.permeability[0], coefficient


def solve_at_stress(graph, responses, stress_MPa, reference_MPa, side="right"):
    """Update g_e = g_e(ref) K_class(stress)/K_class(ref), then solve.

    Graph conductances must correspond to reference_MPa. Returns the solution
    and one local log-sensitivity coefficient per original edge.
    """
    missing = set(graph.response_classes) - responses.keys()
    if missing:
        raise ModelInputError(f"Missing responses: {sorted(missing)}")
    if stress_MPa < reference_MPa:
        raise ModelInputError("First-loading stress must be at least the reference stress")
    values = {}
    for name in set(graph.response_classes):
        response = responses[name]
        value, coefficient = response.at(stress_MPa, side)
        reference, _ = response.at(reference_MPa, side)
        values[name] = (value/reference, coefficient)
    ratios = np.array([values[name][0] for name in graph.response_classes])
    coefficients = np.array([values[name][1] for name in graph.response_classes])
    return solve(graph, graph.g0_m3_Pa_s*ratios), coefficients


def graph_from_dict(item):
    """Load an explicit graph from the documented JSON example schema."""
    return Graph(**{name: item[name] for name in (
        "coordinates_m", "edges", "boundary", "g0_m3_Pa_s", "response_classes",
        "fracture_ids", "volume_m3", "viscosity_Pa_s")}, metadata=item.get("metadata", {}))


def run_case(case):
    """Run an explicit graph and first-loading path; return JSON-safe results."""
    graph = graph_from_dict(case["graph"])
    responses = {name: LogLinearResponse(name, item["stress_MPa"], item["permeability"])
                 for name, item in case["responses"].items()}
    reference = float(case["reference_MPa"])
    path = np.asarray(case["stress_path_MPa"], dtype=float)
    if path.ndim != 1 or not len(path) or not np.all(np.isfinite(path)) or path[0] != reference or np.any(np.diff(path) < 0):
        raise ModelInputError("Path must start at the reference and increase monotonically")
    directions = case.get("directions", {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]})
    states = []
    for stress in path:
        solution, coefficients = solve_at_stress(graph, responses, float(stress), reference)
        states.append({"stress_MPa": float(stress), "K_m2": solution.permeability_m2.tolist(),
                       "dK_dMPa": solution.derivative(coefficients).tolist(),
                       "directions": {name: solution.directional(direction, coefficients)
                                      for name, direction in directions.items()},
                       "diagnostics": solution.diagnostics})
    return {"schema": "coal-sensitive-results-v1", "data_kind": case.get("data_kind", "unspecified"),
            "reference_MPa": reference, "derivative_side": "right (inward at upper endpoint)", "states": states}


def main():
    """Run ``python -m coal_sensitive case.json --output results.json``."""
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, help="Graph and local response JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Results JSON path")
    args = parser.parse_args()
    result = run_case(json.loads(args.case.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Wrote {len(result['states'])} states to {args.output}")


if __name__ == "__main__":
    main()
