"""
verify_cpp_outputs.py -- compare the C++ benchmark's ONNX Runtime outputs with PyTorch for every grid model.

Usage:
    LocalSequenceGameOptimizationCores.exe --verify-outputs cpp_lp0 --verify-lp 0      (P-core LP)
    LocalSequenceGameOptimizationCores.exe --verify-outputs cpp_lp12 --verify-lp 12    (E-core LP)
    python verify_cpp_outputs.py --models models_latest --cpp cpp_lp0 cpp_lp12 --out cpp_output_check.csv

The C++ harness (--verify-outputs) runs each of the 60 grid models at 50, 100 and 200 agents with the timed
session options, tensor layout, probe run and warm-up batches, on the fixed benchmark inputs and on three
hash-generated inputs, and writes the outputs of the first measured batch. This script

  * rebuilds each model in PyTorch from the exported ONNX weights (the trained PyTorch weights were not saved)
    and checks the rebuild against ONNX Runtime in Python on one input per model;
  * regenerates the same inputs bit for bit (checked against the C++ input checksums);
  * compares the C++ outputs with the PyTorch float64 forward pass (the reference) and reports the PyTorch
    float32 error beside it for scale;
  * with two --cpp folders, checks that the outputs are bitwise identical on both LPs.

A case passes when |cpp - ref| <= ATOL + RTOL * |ref| for every output, every output is finite, and the probe
and first measured batch gave bitwise identical outputs.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from onnx import numpy_helper
from torch import nn

SEQ_LEN, OUT = 10, 3
ATOL, RTOL = 1e-5, 1e-4


def hash_uniform(n, seed, lo, hi):
    """Same hash as hashUniform() in steady_state_compute_bound.cpp (uint32 arithmetic wraps)."""
    i = np.arange(n, dtype=np.uint32)
    with np.errstate(over="ignore"):
        h = i * np.uint32(0x9E3779B1) + np.uint32((seed * 0x85EBCA77) & 0xFFFFFFFF)
        h ^= h >> np.uint32(16)
        h *= np.uint32(0x7FEB352D)
        h ^= h >> np.uint32(15)
        h *= np.uint32(0x846CA68B)
        h ^= h >> np.uint32(16)
    u = (h >> np.uint32(8)).astype(np.float32) * np.float32(1.0 / 16777216.0)
    return np.float32(lo) + np.float32(hi - lo) * u


def bench_inputs(n):
    """Same values as fillBenchmarkInputs() in steady_state_compute_bound.cpp."""
    x = np.full(n, 0.5, dtype=np.float32)
    idx = np.arange(0, n, 16)
    x[idx] = np.float32(0.5) + (idx % 100).astype(np.float32) * np.float32(0.001)
    return x


class GRUPredictor(nn.Module):
    """The forward pass of path_predicting_training.py: zero h0, GRU, linear head on the last step."""

    def __init__(self, inputs, hidden, layers):
        super().__init__()
        self.gru = nn.GRU(inputs, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, OUT)

    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size, dtype=x.dtype)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])


def rebuild_from_onnx(path, inputs, hidden, layers):
    """PyTorch module with the exported weights. ONNX orders the gates (z, r, h); PyTorch orders them (r, z, n)."""
    m = onnx.load(str(path))
    init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    grus = [n for n in m.graph.node if n.op_type == "GRU"]
    gemm = [n for n in m.graph.node if n.op_type == "Gemm"]
    assert len(grus) == layers and len(gemm) == 1, f"{path.name}: {len(grus)} GRU nodes, {len(gemm)} Gemm nodes"
    H = hidden
    reorder = lambda a: np.concatenate([a[H:2 * H], a[0:H], a[2 * H:3 * H]])
    model = GRUPredictor(inputs, hidden, layers)
    with torch.no_grad():
        for k, node in enumerate(grus):
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            assert attrs.get("linear_before_reset") == 1 and attrs.get("hidden_size") == H
            W, R, B = (init[node.input[j]][0] for j in (1, 2, 3))
            getattr(model.gru, f"weight_ih_l{k}").copy_(torch.from_numpy(reorder(W)))
            getattr(model.gru, f"weight_hh_l{k}").copy_(torch.from_numpy(reorder(R)))
            getattr(model.gru, f"bias_ih_l{k}").copy_(torch.from_numpy(reorder(B[:3 * H])))
            getattr(model.gru, f"bias_hh_l{k}").copy_(torch.from_numpy(reorder(B[3 * H:])))
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in gemm[0].attribute}
        assert attrs.get("transB") == 1
        model.fc.weight.copy_(torch.from_numpy(init[gemm[0].input[1]]))
        model.fc.bias.copy_(torch.from_numpy(init[gemm[0].input[2]]))
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="folder with the exported ONNX models")
    ap.add_argument("--cpp", nargs="+", required=True, help="one or more --verify-outputs folders")
    ap.add_argument("--out", required=True, help="per-case CSV")
    args = ap.parse_args()

    cpp_dirs = [Path(d) for d in args.cpp]
    man = pd.read_csv(cpp_dirs[0] / "verify_manifest.csv")
    others = [pd.read_csv(d / "verify_manifest.csv") for d in cpp_dirs[1:]]
    rows, rebuild = [], []
    for (I, H, L), cases in man.groupby(["inputs", "hidden", "layers"], sort=False):
        I, H, L = int(I), int(H), int(L)
        onnx_path = Path(args.models) / f"pathing_model_in{I}_H{H}_L{L}.onnx"
        m32 = rebuild_from_onnx(onnx_path, I, H, L)
        m64 = rebuild_from_onnx(onnx_path, I, H, L).double()

        # The rebuild must reproduce ONNX Runtime (Python) on the exported graph.
        x = hash_uniform(50 * SEQ_LEN * I, 1, -1.0, 1.0).reshape(50, SEQ_LEN, I)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        y_ort = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        with torch.no_grad():
            y_t = m32(torch.from_numpy(x)).numpy()
        rebuild.append(float(np.abs(y_ort - y_t).max()))

        for _, c in cases.iterrows():
            n = int(c.agents) * SEQ_LEN * I
            x = bench_inputs(n) if c.input_set == "bench" else hash_uniform(n, int(c.seed), float(c.lo), float(c.hi))
            bits_ok = int(x.view(np.uint32).astype(np.uint64).sum()) == int(c.input_bits_sum)
            x = x.reshape(int(c.agents), SEQ_LEN, I)
            with torch.no_grad():
                ref = m64(torch.from_numpy(x.astype(np.float64))).numpy()
                t32 = m32(torch.from_numpy(x)).numpy().astype(np.float64)
            cpp = np.fromfile(cpp_dirs[0] / c.file, dtype=np.float32).astype(np.float64).reshape(ref.shape)
            err = np.abs(cpp - ref)
            same_lp = all(
                np.array_equal(np.fromfile(d / c.file, dtype=np.uint32), np.fromfile(cpp_dirs[0] / c.file, dtype=np.uint32))
                for d in cpp_dirs[1:])
            rows.append(dict(
                inputs=I, hidden=H, layers=L, agents=int(c.agents), input_set=c.input_set, inputs_match=bits_ok,
                max_abs_ref=float(np.abs(ref).max()), cpp_max_abs_err=float(err.max()),
                cpp_max_rel_err=float((err / np.maximum(np.abs(ref), 1e-12)).max()),
                torch32_max_abs_err=float(np.abs(t32 - ref).max()),
                within_tol=bool((err <= ATOL + RTOL * np.abs(ref)).all()), all_finite=bool(c.all_finite),
                repeat_identical=bool(c.repeat_identical), identical_across_lps=bool(same_lp)))
        print(f"in{I}_H{H}_L{L}: max |cpp - ref| = {max(r['cpp_max_abs_err'] for r in rows[-len(cases):]):.2e}")

    df = pd.DataFrame(rows)
    df["passed"] = df.inputs_match & df.within_tol & df.all_finite & df.repeat_identical
    df.to_csv(args.out, index=False)
    print(f"\nONNX Runtime (C++) {open(cpp_dirs[0] / 'verify_info.txt').readline().split()[-1]}; "
          f"reference PyTorch {torch.__version__} float64; rebuild vs ONNX Runtime {ort.__version__} (Python): "
          f"max |diff| {max(rebuild):.2e} over {len(rebuild)} models")
    print(f"cases {len(df)} ({df[['inputs', 'hidden', 'layers']].drop_duplicates().shape[0]} models x "
          f"{df.agents.nunique()} agent counts x {df.input_set.nunique()} inputs); passed {int(df.passed.sum())}")
    print(f"inputs regenerated bit for bit: {int(df.inputs_match.sum())}/{len(df)}; finite: {int(df.all_finite.sum())}; "
          f"probe = measured batch bitwise: {int(df.repeat_identical.sum())}")
    print(f"C++ max abs error {df.cpp_max_abs_err.max():.2e} (median {df.cpp_max_abs_err.median():.2e}); "
          f"PyTorch float32 max abs error {df.torch32_max_abs_err.max():.2e}; largest |output| {df.max_abs_ref.max():.3g}")
    worst = df.loc[df.cpp_max_abs_err.idxmax()]
    print(f"largest error: in{worst.inputs}_H{worst.hidden}_L{worst.layers} A={worst.agents} {worst.input_set}")
    if others:
        print(f"bitwise identical across {len(cpp_dirs)} LP runs: {int(df.identical_across_lps.sum())}/{len(df)}")


if __name__ == "__main__":
    main()
