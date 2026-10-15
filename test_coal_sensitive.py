"""Analytical, derivative, invariance, calibration and voxel equivalence tests."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose
from coal_sensitive import (Graph, ModelInputError, LogLinearResponse, SeparatedResponse,
    PhasePrior, CondensedVoxel, solve, solve_at_stress, graph_from_labels,
    graph_from_dict, porous_conductance, fracture_conductance, run_case)

ROOT = Path(__file__).resolve().parent


def line_graph():
    return Graph([[0,0,0],[.02,0,0],[.04,0,0]], [[0,1],[1,2]], [1,0,1],
                 [2e-10,5e-10], ("fracture","background"), ("F1",None), 1e-4, 1e-3)


def laws():
    return {name: LogLinearResponse(name,[3.7,10.,20.], np.exp(-c*(np.array([3.7,10.,20.])-3.7)))
            for name,c in (("fracture",.3),("background",.1))}


def heterogeneous():
    labels=np.zeros((4,5,6),dtype=int)
    labels[:,2,:]=1
    labels[1,:,3]=2
    phases={0:PhasePrior("background",2e-14,None,"synthetic background"),
            1:PhasePrior("fracture",8e-13,"F1","synthetic fracture 1"),
            2:PhasePrior("fracture",3e-13,"F2","synthetic fracture 2")}
    return labels,phases


class AnalyticalTests(unittest.TestCase):
    def test_series_permeability_and_sensitivity(self):
        graph=line_graph(); sol=solve(graph)
        expected=graph.viscosity_Pa_s/graph.volume_m3*.04**2/(1/2e-10+1/5e-10)
        self.assertAlmostEqual(sol.permeability_m2[0,0]/expected,1,places=13)
        metric=sol.directional([1,0,0],np.array([.3,.1]))
        self.assertAlmostEqual(metric["S_per_MPa"],(5*.3+2*.1)/7,places=13)
        assert_allclose(metric["edge_weights"],[5/7,2/7],rtol=1e-13)

    def test_parallel_permeability_and_sensitivity(self):
        graph=Graph([[0,0,0],[.04,0,0]],[[0,1],[0,1]],[1,1],
                    [2e-10,5e-10],("fracture","background"),("F1",None),1e-4,1e-3)
        sol=solve(graph); metric=sol.directional([1,0,0],[.3,.1])
        self.assertAlmostEqual(sol.permeability_m2[0,0]/(10*.04**2*7e-10),1,places=13)
        self.assertAlmostEqual(metric["S_per_MPa"],(2*.3+5*.1)/7,places=13)
        self.assertEqual(sol.diagnostics["factorizations"],0)

    def test_isotropic_star(self):
        r=.025; g=2e-10; mu=1e-3; volume=1e-4
        xyz=np.vstack([np.zeros((1,3)),r*np.eye(3),-r*np.eye(3)])
        graph=Graph(xyz,[(0,i) for i in range(1,7)],[0,1,1,1,1,1,1],
                    [g]*6,("background",)*6,(None,)*6,volume,mu)
        assert_allclose(solve(graph).permeability_m2,np.eye(3)*2*g*r*r*mu/volume,rtol=1e-13,atol=1e-28)

    def test_homogeneous_anisotropic_voxel_spacing(self):
        graph=graph_from_labels(np.zeros((3,4,5),int),[.001,.002,.003],
            {0:PhasePrior("background",2e-14,None,"synthetic homogeneous")},1e-3)
        assert_allclose(solve(graph).permeability_m2,np.eye(3)*2e-14,rtol=1e-12,atol=1e-28)

    def test_conductance_formulas(self):
        self.assertAlmostEqual(porous_conductance(2e-14,.001,.02,.001)/1e-12,1)
        self.assertAlmostEqual(fracture_conductance(1e-4,.01,.02,.001)/(0.01*(1e-4)**3/(12*.001*.02)),1)


class NetworkTests(unittest.TestCase):
    def test_derivative_by_finite_difference(self):
        graph=line_graph(); response=laws(); s=8.; h=1e-4
        center,c=solve_at_stress(graph,response,s,3.7)
        plus,_=solve_at_stress(graph,response,s+h,3.7)
        minus,_=solve_at_stress(graph,response,s-h,3.7)
        assert_allclose(center.derivative(c),(plus.permeability_m2-minus.permeability_m2)/(2*h),rtol=2e-8,atol=1e-26)

    def test_contribution_closure_and_updated_weights(self):
        metrics=[]
        for s in (3.7,20.):
            sol,c=solve_at_stress(line_graph(),laws(),s,3.7)
            m=sol.directional([1,0,0],c); metrics.append(m)
            self.assertAlmostEqual(sum(m["edge_weights"]),1)
            self.assertAlmostEqual(m["S_per_MPa"],m["background_contribution_per_MPa"]+sum(v["J_per_MPa"] for v in m["fracture_contributions"].values()))
            self.assertGreaterEqual(m["S_per_MPa"],min(c)-1e-14)
            self.assertLessEqual(m["S_per_MPa"],max(c)+1e-14)
        self.assertGreater(metrics[1]["edge_weights"][0],metrics[0]["edge_weights"][0])

    def test_translation_rotation_edge_orientation(self):
        labels,phases=heterogeneous()
        graph=graph_from_labels(labels,[.001]*3,phases,1e-3)
        angle=.43; R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        changed=replace(graph,coordinates_m=graph.coordinates_m@R.T+[.1,.2,.3],edges=graph.edges[:,::-1])
        assert_allclose(solve(changed).permeability_m2,R@solve(graph).permeability_m2@R.T,rtol=1e-10,atol=1e-26)

    def test_floating_component_has_zero_flow(self):
        base=line_graph()
        graph=Graph(np.vstack([base.coordinates_m,[[0,.1,0],[0,.2,0]]]),
             [[0,1],[1,2],[3,4]],[1,0,1,0,0],[2e-10,5e-10,3e-10],("background",)*3,(None,)*3,1e-4,1e-3)
        sol=solve(graph)
        assert_allclose(sol.permeability_m2,solve(base).permeability_m2,rtol=1e-13)
        self.assertEqual(sol.diagnostics["floating_components"],1)
        assert_allclose(sol.edge_coordinate_drop[-1],[0,0,0],atol=0)

    def test_zero_permeability_status(self):
        graph=replace(line_graph(),g0_m3_Pa_s=np.zeros(2))
        metric=solve(graph).directional([1,0,0],[.3,.1])
        self.assertEqual(metric["status"],"zero_directional_permeability")
        self.assertIsNone(metric["S_per_MPa"])
        json.dumps(metric,allow_nan=False)

    def test_uniform_scale_changes_k_not_sensitivity(self):
        graph=line_graph(); c=[.3,.1]
        a=solve(graph); b=solve(graph,graph.g0_m3_Pa_s*7)
        assert_allclose(b.permeability_m2,7*a.permeability_m2,rtol=1e-13)
        self.assertAlmostEqual(a.directional([1,0,0],c)["S_per_MPa"],b.directional([1,0,0],c)["S_per_MPa"])

    def test_invalid_graph_and_direction(self):
        for change in ({"g0_m3_Pa_s":[-1,1]}, {"edges":[[0,.5],[1,2]]}, {"boundary":[0,0,0]}):
            with self.subTest(change=change),self.assertRaises(ModelInputError): replace(line_graph(),**change)
        with self.assertRaises(ModelInputError): solve(line_graph()).directional([0,0,0],[.3,.1])


class ResponseTests(unittest.TestCase):
    def test_exact_knots_and_log_derivative(self):
        r=LogLinearResponse("sample",[3.7,10,20],[100,50,10])
        self.assertAlmostEqual(r.at(10)[0],50)
        self.assertAlmostEqual(r.at(6)[1],np.log(2)/6.3)
        self.assertAlmostEqual(r.at(10,"left")[1],np.log(2)/6.3)
        self.assertAlmostEqual(r.at(10,"right")[1],np.log(5)/10)

    def test_response_range_and_input_checks(self):
        r=laws()["background"]
        for stress in (3.,21.,np.nan):
            with self.subTest(stress=stress),self.assertRaises(ModelInputError): r.at(stress)
        for x,k in (([3.7,3.7],[2,1]),([3.7,20],[1,2]),([3.7,20],[1,0])):
            with self.assertRaises(ModelInputError): LogLinearResponse("bad",x,k)
        with self.assertRaises(ModelInputError): r.evaluate(10,"unloading")

    def test_parallel_component_reconstruction(self):
        bulk=LogLinearResponse("bulk",[3.7,10,20],[100,50,20])
        background=LogLinearResponse("background",[3.7,10,20],[10,8,5])
        separated=SeparatedResponse(bulk,background,.1)
        for s in np.linspace(3.7,20,25):
            k,c=separated.at(s); kb,cb=background.at(s); kt,ct=bulk.at(s)
            self.assertAlmostEqual((.1*k+.9*kb)/kt,1,places=12)
            self.assertAlmostEqual((.1*k*c+.9*kb*cb)/(kt*ct),1,places=12)
        h=1e-4; s=8.
        fd=-(np.log(separated.at(s+h)[0])-np.log(separated.at(s-h)[0]))/(2*h)
        self.assertAlmostEqual(fd,separated.at(s)[1],places=9)

    def test_incompatible_component_calibration(self):
        bulk=LogLinearResponse("bulk",[3.7,20],[1,.5])
        background=LogLinearResponse("background",[3.7,20],[10,5])
        with self.assertRaises(ModelInputError): SeparatedResponse(bulk,background,.1)
        with self.assertRaises(ModelInputError): SeparatedResponse(bulk,bulk,0)

    def test_common_reference_rebasing(self):
        sol,c=solve_at_stress(line_graph(),laws(),10,10)
        assert_allclose(sol.permeability_m2,solve(line_graph()).permeability_m2,rtol=1e-13)
        with self.assertRaises(ModelInputError): solve_at_stress(line_graph(),{},10,3.7)
        with self.assertRaises(ModelInputError): solve_at_stress(line_graph(),laws(),5,10)


class VoxelTests(unittest.TestCase):
    def test_condensed_matches_full_graph_and_contributions(self):
        labels,phases=heterogeneous(); responses=laws()
        graph=graph_from_labels(labels,[.001]*3,phases,1e-3)
        for method in ("direct","cg"):
            condensed=CondensedVoxel(labels,.001,phases,method=method)
            for s in (3.7,8.,20.):
                with self.subTest(method=method,stress=s):
                    full,c=solve_at_stress(graph,responses,s,3.7)
                    fast=condensed.solve(responses,s,"L")
                    assert_allclose(fast["K_m2"],full.permeability_m2,rtol=1e-8,atol=1e-25)
                    assert_allclose(fast["dK_dMPa_right"],full.derivative(c),rtol=1e-8,atol=1e-25)
                    for axis,d in enumerate("xyz"):
                        m=full.directional(np.eye(3)[axis],c); f=fast["directions"][d]
                        self.assertAlmostEqual(m["S_per_MPa"],f["S_right_per_MPa"],places=10)
                        for owner,value in m["fracture_contributions"].items():
                            self.assertAlmostEqual(value["J_per_MPa"],f["fracture_contributions_right"][owner]["J_per_MPa"],places=10)

    def test_condensed_homogeneous_solution(self):
        phases={0:PhasePrior("background",2e-14,None,"synthetic")}
        result=CondensedVoxel(np.zeros((3,4,5),int),.001,phases,method="direct").solve(laws(),3.7,"L")
        assert_allclose(result["K_m2"],np.eye(3)*2e-14,rtol=1e-12,atol=1e-28)

    def test_voxel_derivative_finite_difference(self):
        labels,phases=heterogeneous(); solver=CondensedVoxel(labels,.001,phases,method="direct")
        responses=laws(); s=8.; h=1e-4
        center=solver.solve(responses,s,"L")
        plus=np.array(solver.solve(responses,s+h,"L")["K_m2"])
        minus=np.array(solver.solve(responses,s-h,"L")["K_m2"])
        assert_allclose(center["dK_dMPa_right"],(plus-minus)/(2*h),rtol=1e-7,atol=1e-25)

    def test_voxel_label_validation(self):
        phases={0:PhasePrior("background",2e-14,None,"synthetic")}
        with self.assertRaises(ModelInputError): graph_from_labels(np.ones((2,2,2),int),[.001]*3,phases,1e-3)
        with self.assertRaises(ModelInputError): graph_from_labels(np.zeros((2,2,2)),[.001]*3,phases,1e-3)
        with self.assertRaises(ValueError): CondensedVoxel(np.full((2,2,2),-1),.001,phases)


class FileTests(unittest.TestCase):
    def test_example_against_independent_series_formula(self):
        case=json.loads((ROOT/"example_case.json").read_text())
        result=run_case(case)
        for state in result["states"]:
            s=state["stress_MPa"]-3.7
            g1,g2=2e-10*np.exp(-.3*s),5e-10*np.exp(-.1*s)
            expected=10*.04**2/(1/g1+1/g2)
            self.assertAlmostEqual(state["directions"]["x"]["K_m2"]/expected,1,places=12)
            self.assertAlmostEqual(state["directions"]["x"]["S_per_MPa"],(g2*.3+g1*.1)/(g1+g2),places=12)
        json.dumps(result,allow_nan=False)

    def test_example_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"results.json"
            proc=subprocess.run([sys.executable,"-m","coal_sensitive",str(ROOT/"example_case.json"),"--output",str(output)],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertEqual(len(json.loads(output.read_text())["states"]),5)

    def test_invalid_loading_path(self):
        case=json.loads((ROOT/"example_case.json").read_text())
        case["stress_path_MPa"]=[3.7,10,8]
        with self.assertRaises(ModelInputError): run_case(case)


if __name__ == "__main__":
    unittest.main()
