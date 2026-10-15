"""Run five small, deterministic cases and save a summary under results/."""
import argparse
import json
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from coal_sensitive import (Graph, PhasePrior, CondensedVoxel, LogLinearResponse,
                            SeparatedResponse, graph_from_labels, run_case, solve)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("results/verification.json"))
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    case=json.loads((root/"example_case.json").read_text())
    series=run_case(case)
    expected=json.loads((root/"expected_series.json").read_text())
    for state,target in zip(series["states"],expected["states"]):
        metric=state["directions"]["x"]
        assert_allclose([metric["K_m2"],metric["S_per_MPa"]],
                       [target["K_x_m2"],target["S_x_per_MPa"]],rtol=expected["relative_tolerance"],atol=0)
    graph=Graph([[0,0,0],[.04,0,0]],[[0,1],[0,1]],[1,1],[2e-10,5e-10],
                ("fracture","background"),("F1",None),1e-4,1e-3)
    parallel=solve(graph).directional([1,0,0],[.3,.1])
    assert_allclose([parallel["K_m2"],parallel["S_per_MPa"]],
                   [10*.04**2*7e-10,(2*.3+5*.1)/7],rtol=1e-12,atol=0)
    phases={0:PhasePrior("background",2e-14,None,"synthetic background"),
            1:PhasePrior("fracture",8e-13,"F1","synthetic fracture plane"),
            2:PhasePrior("fracture",3e-13,"F2","synthetic intersecting segment")}
    homogeneous=solve(graph_from_labels(np.zeros((4,5,6),int),[.001]*3,phases,1e-3))
    assert_allclose(homogeneous.permeability_m2,np.eye(3)*2e-14,rtol=1e-12,atol=1e-28)
    labels=np.zeros((4,5,6),int); labels[:,2,:]=1; labels[1,:,3]=2
    responses={name:LogLinearResponse(name,item["stress_MPa"],item["permeability"])
               for name,item in case["responses"].items()}
    voxel=CondensedVoxel(labels,.001,phases,method="direct")
    voxel_states=[voxel.solve(responses,s,"L") for s in (3.7,10,20)]
    full=solve(graph_from_labels(labels,[.001]*3,phases,1e-3))
    assert_allclose(voxel_states[0]["K_m2"],full.permeability_m2,rtol=1e-9,atol=1e-25)
    bulk=LogLinearResponse("bulk",[3.7,10,20],[100,50,20])
    background=LogLinearResponse("background",[3.7,10,20],[10,8,5])
    fracture=SeparatedResponse(bulk,background,.1)
    separation=[]
    for stress in (3.7,8,10,15,20):
        k,c=fracture.at(stress); kb,_=background.at(stress); kt,_=bulk.at(stress)
        assert_allclose(.1*k+.9*kb,kt,rtol=1e-12)
        separation.append({"stress_MPa":stress,"fracture_permeability_in_input_unit":k,"sensitivity_per_MPa":c})
    output={"data_kind":"synthetic verification cases","checks_passed":True,
            "series":series,"parallel":parallel,"homogeneous_K_m2":homogeneous.permeability_m2.tolist(),
            "heterogeneous_voxel":voxel_states,"component_separation":separation}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print("PASS: series, parallel, homogeneous volume, heterogeneous volume, component separation")
    print(f"Results: {args.output}")


if __name__=="__main__":
    main()
