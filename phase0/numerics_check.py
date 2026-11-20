"""Why does bare Samudra on CUDA differ from fme's captured output (0.0136) and batch4 vs single (0.016)?"""
import json, os, sys, numpy as np, torch
from hc import graph
S = sys.argv[1]
ocn, ins, outs, means, stds = graph.load_ocean()
x = torch.tensor(graph.load_state(0, S)[None], device="cuda")
ycap = torch.tensor(np.load(os.path.join(S, "y_000.npy")), device="cuda")
isst = outs.index("sst")
res = {"defaults": {"cudnn.allow_tf32": torch.backends.cudnn.allow_tf32, "matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn.benchmark": torch.backends.cudnn.benchmark}}
def run(tag):
    with torch.no_grad():
        y1 = ocn(x.clone())[0]; y1b = ocn(x.clone())[0]
        y4 = ocn(x.expand(4, -1, -1, -1).clone())[2]
    res[tag] = {"vs_fme_capture_maxabs": float((y1 - ycap).abs().max()),
                "vs_fme_capture_sst_maxabs_K": float((y1 - ycap)[isst].abs().max()) * stds["sst"],
                "repeat_maxabs": float((y1 - y1b).abs().max()),
                "batch4_vs_single_maxabs": float((y4 - y1).abs().max()),
                "batch4_vs_single_sst_maxabs_K": float((y4 - y1)[isst].abs().max()) * stds["sst"]}
run("tf32_default")
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
run("tf32_off")
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
run("tf32_off_deterministic")
json.dump(res, open("phase0/numerics_check.json", "w"), indent=1); print(json.dumps(res, indent=1))
