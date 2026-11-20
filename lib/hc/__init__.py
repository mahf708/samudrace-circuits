"""hackathon-circuits shared library: paths, exact Samudra graph/replay, law helpers."""
import os


def set_numerics():
    """Perturbation/patching/JVP experiments run in true fp32 (TF32 off; HC_TF32=1 to keep the CUDA default).
    PyTorch enables TF32 for cuDNN convolutions by default and fme runs that way. With TF32 the U-Net is not
    batch-invariant (batch-of-4 vs single differs up to 0.021 K SST); fp32 is (4e-5 K) and differs from the
    deployed TF32 output by at most 0.06 K pointwise (phase0/numerics_check.json)."""
    import torch

    tf32 = os.environ.get("HC_TF32", "0") == "1"
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    return tf32
