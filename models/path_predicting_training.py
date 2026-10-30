"""
path_predicting_training.py

Builds the GRU path-prediction models used by the hybrid-core placement benchmark
(steady_state_compute_bound.cpp), exports them to ONNX and verifies every export with
ONNX Runtime before it can be benchmarked.

For every (inputs, hidden, layers) in the grid:
  1. seed random / numpy / torch from a per-model seed (reproducible regardless of build order),
  2. train for 2 epochs on synthetic trajectories (only the graph structure matters for latency),
  3. export with the TorchScript exporter (dynamo=False), opset 17, dynamic batch axis,
  4. check the graph (opset, one ONNX GRU node per layer, dynamic batch dimension) and verify it with
     onnxruntime at batch=1 and batch=200 against PyTorch (atol 1e-4),
  5. append a row to <output-dir>/manifest.csv (parameter count, size, SHA-256, versions).

A model that fails any check is renamed to *.onnx.failed (so the C++ benchmark never loads it),
recorded in the manifest with verified=False, and the script stops with an error.

Output: models_latest/pathing_model_in{I}_H{H}_L{L}.onnx and models_latest/manifest.csv.
Copy the whole models_latest folder next to LocalSequenceGameOptimizationCores.exe (x64/Release).
"""

import argparse
import csv
import hashlib
import inspect
import os
import random
import sys
import time
from datetime import datetime

# Must be set before CUDA is initialised; only matters if a CUDA device is ever used (training runs on CPU).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.optim as optim

SEQ_LENGTH = 10          # must match SEQ_LEN in the C++ benchmark
OPSET_VERSION = 17       # ONNX GRU exists since opset 7 (latest revision 14), so opset 17 exports GRU natively
BASE_SEED = 1337
EPOCHS = 2               # only the graph structure matters for latency testing, not accuracy
NUM_SAMPLES = 500
LEARNING_RATE = 0.01
VERIFY_BATCH_SIZES = (1, 200)
VERIFY_ATOL = 1e-4

INPUT_SIZES = [10, 25, 50, 100, 250, 500, 750, 1000]
HIDDEN_SIZES = [32, 64, 128, 256, 512]
LAYER_COUNTS = [1, 2, 4, 8, 16]

MANIFEST_FIELDS = [
    "filename", "inputs", "hidden", "layers", "parameter_count", "file_size_bytes", "sha256",
    "verified", "max_abs_diff_b1", "max_abs_diff_b200", "opset", "gru_nodes", "seed",
    "torch_version", "onnx_version", "onnxruntime_version", "exported_at", "error",
]


class ModelVerificationError(RuntimeError):
    pass


def configure_determinism():
    # Raises if a nondeterministic kernel is used, instead of silently producing different weights.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device


def model_seed(base_seed, inputs, hidden, layers):
    return (base_seed + inputs * 1_000_000 + hidden * 1_000 + layers) % (2 ** 32)


def generate_trajectory_data(num_samples=NUM_SAMPLES, seq_length=SEQ_LENGTH, input_features=16):
    print(f"Generating {num_samples} synthetic trajectories with {input_features} sensory features...")
    # Dummy sensory data so the tensor has the size the C++ benchmark feeds. One vectorised draw yields the
    # same values (C order) as drawing frame by frame.
    data = np.random.uniform(-1.0, 1.0, size=(num_samples, seq_length + 1, input_features))
    X = data[:, :-1, :]
    # We only predict the next X, Y, Z (first 3 features)
    Y = data[:, -1, 0:3]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


class ProductionPathPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=3):
        super(ProductionPathPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # A dynamic GRU based on the profile passed in
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


def train_model(input_features, hidden_size, num_layers):
    X, Y = generate_trajectory_data(num_samples=NUM_SAMPLES, seq_length=SEQ_LENGTH, input_features=input_features)
    model = ProductionPathPredictor(input_size=input_features, hidden_size=hidden_size, num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        model.train()
        outputs = model(X)
        loss = criterion(outputs, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f}")

    model.eval()
    return model


def export_to_onnx(model, input_features, hidden_size, num_layers, output_dir):
    onnx_file_path = os.path.join(output_dir, f"pathing_model_in{input_features}_H{hidden_size}_L{num_layers}.onnx")
    print(f"Exporting to {onnx_file_path}...")
    model.eval()

    dummy_input = torch.randn(1, SEQ_LENGTH, input_features)
    export_kwargs = dict(
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    if "dynamo" not in inspect.signature(torch.onnx.export).parameters:
        raise RuntimeError(f"torch {torch.__version__} has no 'dynamo' argument on torch.onnx.export; use torch >= 2.5")
    # Explicitly the TorchScript-based exporter (newer torch versions default to dynamo=True).
    # It prints a UserWarning about GRU + batch_size != 1 whenever h0 is not a graph input. Here h0 is built
    # from x.size(0) inside the graph, and verify_with_onnxruntime() checks batch=1 and batch=200 explicitly.
    torch.onnx.export(model, (dummy_input,), onnx_file_path, dynamo=False, **export_kwargs)
    print(f"Successfully exported to: {onnx_file_path}")
    return onnx_file_path


def inspect_onnx_graph(onnx_path):
    """Returns (opset, number of GRU nodes, batch dimension is dynamic)."""
    onnx.checker.check_model(onnx_path)
    model = onnx.load(onnx_path)
    opset = next((imp.version for imp in model.opset_import if imp.domain in ("", "ai.onnx")), None)
    gru_nodes = sum(1 for node in model.graph.node if node.op_type == "GRU")
    batch_dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    dynamic_batch = bool(batch_dim.dim_param) or not batch_dim.HasField("dim_value")
    return opset, gru_nodes, dynamic_batch


def verify_with_onnxruntime(model, onnx_path, input_features, seed):
    """Runs batch=1 and batch=200 through ONNX Runtime and PyTorch; returns {batch: max_abs_diff}."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # same as the C++ benchmark
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(seed)
    diffs = {}
    for batch in VERIFY_BATCH_SIZES:
        x = rng.uniform(-1.0, 1.0, size=(batch, SEQ_LENGTH, input_features)).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()
        try:
            actual = session.run(["output"], {"input": x})[0]
        except Exception as e:
            raise ModelVerificationError(f"onnxruntime failed at batch={batch}: {e}") from e
        if actual.shape != (batch, 3):
            raise ModelVerificationError(f"batch={batch}: output shape {actual.shape}, expected {(batch, 3)}")
        max_abs = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, rtol=0.0, atol=VERIFY_ATOL):
            raise ModelVerificationError(f"batch={batch}: max |ORT - PyTorch| = {max_abs:.3e} exceeds atol {VERIFY_ATOL:g}")
        diffs[batch] = max_abs
        print(f"  verified batch={batch}: output {actual.shape}, max |ORT - PyTorch| = {max_abs:.3e}")
    return diffs


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_and_verify(inputs, hidden, layers, output_dir, base_seed, writer, manifest_file):
    seed = model_seed(base_seed, inputs, hidden, layers)
    seed_everything(seed)
    model = train_model(inputs, hidden, layers)
    onnx_path = export_to_onnx(model, inputs, hidden, layers, output_dir)

    row = {
        "filename": os.path.basename(onnx_path),
        "inputs": inputs,
        "hidden": hidden,
        "layers": layers,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "file_size_bytes": os.path.getsize(onnx_path),
        "sha256": sha256_of(onnx_path),
        "verified": False,
        "max_abs_diff_b1": "",
        "max_abs_diff_b200": "",
        "opset": "",
        "gru_nodes": "",
        "seed": seed,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "error": "",
    }
    try:
        opset, gru_nodes, dynamic_batch = inspect_onnx_graph(onnx_path)
        row["opset"], row["gru_nodes"] = opset, gru_nodes
        if opset != OPSET_VERSION:
            raise ModelVerificationError(f"exported opset {opset}, expected {OPSET_VERSION}")
        if gru_nodes != layers:
            raise ModelVerificationError(f"graph has {gru_nodes} ONNX GRU nodes, expected one per layer ({layers})")
        if not dynamic_batch:
            raise ModelVerificationError("input batch dimension is not dynamic")
        diffs = verify_with_onnxruntime(model, onnx_path, inputs, seed)
        row["max_abs_diff_b1"] = f"{diffs[1]:.3e}"
        row["max_abs_diff_b200"] = f"{diffs[200]:.3e}"
        row["verified"] = True
    except Exception as e:
        failed_path = onnx_path + ".failed"
        os.replace(onnx_path, failed_path)
        row["filename"] = os.path.basename(failed_path)
        row["error"] = str(e).replace("\n", " ")
        writer.writerow(row)
        manifest_file.flush()
        print("\n" + "!" * 78, file=sys.stderr)
        print(f"VERIFICATION FAILED: {os.path.basename(onnx_path)}: {e}", file=sys.stderr)
        print(f"Model moved to {failed_path}; build stopped.", file=sys.stderr)
        print("!" * 78, file=sys.stderr)
        raise ModelVerificationError(f"{os.path.basename(onnx_path)}: {e}") from e

    writer.writerow(row)
    manifest_file.flush()


def main():
    parser = argparse.ArgumentParser(description="Train, export (ONNX opset 17) and verify the GRU pathing models.")
    parser.add_argument("--output-dir", default="models_latest", help="output folder (default: models_latest)")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help=f"base seed (default: {BASE_SEED})")
    args = parser.parse_args()

    configure_determinism()
    print(f"torch {torch.__version__} | onnx {onnx.__version__} | onnxruntime {ort.__version__} | "
          f"opset {OPSET_VERSION} | base seed {args.seed}")

    # --- SMART FOLDER MANAGEMENT ---
    latest_dir = os.path.normpath(args.output_dir)
    if os.path.exists(latest_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = os.path.join(os.path.dirname(os.path.abspath(latest_dir)), f"models_archive_{timestamp}")
        os.rename(latest_dir, archive_dir)
        print(f"Archived previous models to {archive_dir}")
    os.makedirs(latest_dir)

    # Create a 3D scaling matrix: Inputs x Hidden Size x Layers
    total_models = len(INPUT_SIZES) * len(HIDDEN_SIZES) * len(LAYER_COUNTS)
    current_model = 1
    start = time.time()

    manifest_path = os.path.join(latest_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        manifest_file.flush()

        for inputs in INPUT_SIZES:
            for hidden in HIDDEN_SIZES:
                for layers in LAYER_COUNTS:
                    print("\n===========================================")
                    print(f"Building Model [{current_model}/{total_models}] - Inputs: {inputs}, Hidden: {hidden}, Layers: {layers}")
                    print("===========================================")
                    build_and_verify(inputs, hidden, layers, latest_dir, args.seed, writer, manifest_file)
                    current_model += 1

    print(f"\nAll {total_models} models exported and verified in {time.time() - start:.0f} s. Manifest: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except ModelVerificationError as e:
        print(f"\n[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
