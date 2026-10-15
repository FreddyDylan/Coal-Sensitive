#!/usr/bin/env python3
"""Reproduce the two field-data cases released with the manuscript.

Loads the shared calibration parameters and case files, rebuilds the stress
dependent conductances, solves the condensed voxel network at every stress
state of the first-loading path, and writes result tables in the same format
as the shipped files.  With --verify, the computed values are compared
against the shipped result tables.

Usage:
    cd real_cases
    python run_real_cases.py --output results
    python run_real_cases.py --output results --verify
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from coal_sensitive import CondensedVoxel, LogLinearResponse, SeparatedResponse  # noqa: E402
from coal_sensitive import PhasePrior  # noqa: E402

MD = 9.869233e-16  # m^2 per mD, the constant used by the research pipeline
REFERENCE_MPa = 3.7
MAXIMUM_MPa = 20.0


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_curves(path):
    curves = {}
    for row in read_rows(path):
        curves.setdefault(row["sample"], []).append(
            (float(row["effective_stress_MPa"]), float(row["K_mD"])))
    return {name: (np.array([s for s, _ in rows]), np.array([k for _, k in rows]))
            for name, rows in curves.items()}


def rebased(stress, permeability):
    """Rebase a log-linear first-loading curve to the common reference stress."""
    assert stress[0] <= REFERENCE_MPa
    k_ref = float(np.exp(np.interp(REFERENCE_MPa, stress, np.log(permeability))))
    keep = stress > REFERENCE_MPa
    return (np.r_[REFERENCE_MPa, stress[keep]], np.r_[k_ref, permeability[keep]])


def build_responses(curves, geometry):
    s2, k2 = rebased(*curves["BX-2"])
    s5, k5 = rebased(*curves["BX-5"])
    background = LogLinearResponse("BX-5", s5, k5)
    fraction = geometry["samples"]["BX-2"]["fracture_area_fraction"]
    fracture = SeparatedResponse(LogLinearResponse("BX-2", s2, k2), background, fraction, name="BX-2")
    k0_background = float(k5[0]) * MD
    k0_fracture = float(fracture.permeability[0]) * MD
    responses = {"BX-5": background, "BX-2": fracture}
    return responses, k0_background, k0_fracture


def stress_path(curves):
    knots = {REFERENCE_MPa, MAXIMUM_MPa}
    for name in ("BX-2", "BX-5"):
        knot_stress = curves[name][0]
        knots |= {float(x) for x in knot_stress if REFERENCE_MPa <= x <= MAXIMUM_MPa}
    return sorted(knots)


def run_case(case_dir, responses, k0_background, k0_fracture):
    config = json.loads((case_dir / "case_config.json").read_text(encoding="utf-8"))
    labels = np.load(case_dir / "labels.npy", allow_pickle=False)
    phases = {}
    for label, item in config["phases"].items():
        k0 = k0_background if item["class"] == "BX-5" else k0_fracture
        phases[int(label)] = PhasePrior(item["class"], k0, item["fracture_id"],
                                        "released field-data case")
    solver = CondensedVoxel(labels, config["spacing_m"], phases)
    states = [solver.solve(responses, float(s), "L") for s in config["stress_path_MPa"]]
    initial = {d: states[0]["directions"][d]["K_m2"] for d in "xyz"}
    series, contributions = [], []
    for state in states:
        stress = state["stress_MPa"]
        for direction in ("x", "y", "z"):
            m = state["directions"][direction]
            retention = m["K_m2"] / initial[direction]
            series.append({
                "stress_MPa": stress, "direction": direction, "K_mD": m["K_m2"] / MD,
                "retention": retention, "loss_percent": 100.0 * (1.0 - retention),
                "S_left_per_MPa": m["S_left_per_MPa"], "S_right_per_MPa": m["S_right_per_MPa"],
                "background_left_per_MPa": m["background_left_per_MPa"],
                "background_right_per_MPa": m["background_right_per_MPa"],
                "fracture_energy_fraction": m["fracture_energy_fraction"],
                "N_eff_fractures": m["N_eff_fractures"]})
            for owner in sorted(m["fracture_contributions_left"]):
                left = m["fracture_contributions_left"][owner]
                right = m["fracture_contributions_right"][owner]
                contributions.append({
                    "stress_MPa": stress, "direction": direction, "owner": owner,
                    "energy_weight": left["energy_weight"],
                    "J_left_per_MPa": left["J_per_MPa"],
                    "J_right_per_MPa": right["J_per_MPa"]})
            contributions.append({
                "stress_MPa": stress, "direction": direction, "owner": "background",
                "energy_weight": 1.0 - m["fracture_energy_fraction"],
                "J_left_per_MPa": m["background_left_per_MPa"],
                "J_right_per_MPa": m["background_right_per_MPa"]})
    return config, series, contributions


def write_table(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row[key] is None else repr(row[key])
                                   if isinstance(row[key], float) else row[key])
                             for key in fields})


def _same_key(a, b):
    try:
        return abs(float(a) - float(b)) <= 1e-9 * max(abs(float(a)), abs(float(b)), 1.0)
    except (TypeError, ValueError):
        return str(a) == str(b)


def compare(shipped_rows, computed_rows, fields, keys):
    assert len(shipped_rows) == len(computed_rows), "row count mismatch"
    worst = 0.0
    for a, b in zip(shipped_rows, computed_rows):
        for key in keys:
            assert _same_key(a[key], b[key]), f"row key mismatch: {key}: {a[key]} vs {b[key]}"
        for field in fields:
            if field in keys:
                continue
            x, y = float(a[field]), float(b[field])
            scale = max(abs(x), abs(y), 1e-30)
            worst = max(worst, abs(x - y) / scale)
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true",
                        help="compare the computed tables with the shipped ones")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    curves = load_curves(HERE / "calibration/calibration_curves.csv")
    geometry = json.loads((HERE / "calibration/calibration_geometry.json").read_text(encoding="utf-8"))
    responses, k0_background, k0_fracture = build_responses(curves, geometry)
    path = stress_path(curves)
    print(f"stress path: {path}")
    print(f"k0 background: {k0_background:.15e} m^2 | k0 fracture: {k0_fracture:.15e} m^2")
    print(f"contrast: {k0_fracture / k0_background:.6f}")

    series_fields = ["stress_MPa", "direction", "K_mD", "retention", "loss_percent",
                     "S_left_per_MPa", "S_right_per_MPa", "background_left_per_MPa",
                     "background_right_per_MPa", "fracture_energy_fraction", "N_eff_fractures"]
    contribution_fields = ["stress_MPa", "direction", "owner", "energy_weight",
                           "J_left_per_MPa", "J_right_per_MPa"]
    case_dirs = sorted(path for path in HERE.glob("case_*") if path.is_dir())
    worst_overall = 0.0
    for case_dir in case_dirs:
        config, series, contributions = run_case(case_dir, responses, k0_background, k0_fracture)
        assert abs(config["calibration"]["contrast"] / (k0_fracture / k0_background) - 1) < 1e-9
        assert config["stress_path_MPa"] == path
        tag = case_dir.name
        write_table(args.output / f"{tag}_series.csv", series, series_fields)
        write_table(args.output / f"{tag}_contributions.csv", contributions, contribution_fields)
        final = [row for row in series if row["stress_MPa"] == MAXIMUM_MPa]
        print(f"{tag}: {config['roi_id']} ({config['material']})")
        for row in final:
            print(f"  {row['direction']}: K(20 MPa) = {row['K_mD']:.6f} mD, "
                  f"loss = {row['loss_percent']:.3f} %, S = {row['S_right_per_MPa']:.8f} 1/MPa, "
                  f"fracture energy fraction = {row['fracture_energy_fraction']:.6f}")
        if args.verify:
            for table, fields, keys in (
                    ("series", series_fields, ("stress_MPa", "direction")),
                    ("contributions", contribution_fields, ("stress_MPa", "direction", "owner"))):
                shipped = read_rows(case_dir / f"results_{table}.csv")
                computed = series if table == "series" else contributions
                worst = compare(shipped, computed, fields, keys)
                worst_overall = max(worst_overall, worst)
                print(f"  verify {table}: max relative deviation {worst:.3e}")
    if args.verify:
        print(f"verification passed; worst relative deviation {worst_overall:.3e}")
        assert worst_overall < 1e-9, "reproduction tolerance exceeded"


if __name__ == "__main__":
    main()
