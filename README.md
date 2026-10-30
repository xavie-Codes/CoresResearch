# GRU Placement in a Synthetic Game Loop on a Hybrid Laptop CPU

Code, raw data, and analysis for the paper of the same title by Xavier Gonzalez, Haitham Adarbah, and
Jesus Guillen. The paper source is in [`paper/`](paper/main.tex).

The benchmark runs one gated recurrent unit (GRU) inference thread, standing in for enemy AI, beside a
calibrated 16-ms synthetic game loop on an Intel Core i9-12900H. It places the AI thread on the game
thread's own logical processor (LP), on that P-core's SMT sibling, or on the E-cores, and compares each
placement with a matched no-AI control. Each test records the game's frame times, the AI's batch latency
and missed 50-ms deadlines, and package power.

## Layout

| Folder | Contents |
|---|---|
| [`harness/`](harness) | C++ benchmark (`steady_state_compute_bound.cpp`) and its Visual Studio 2022 solution |
| [`harness/launchers/`](harness/launchers) | Batch files that ran the three sessions (environment checks, services, power plan) |
| [`harness/config/`](harness/config) | `calibrated_loops.txt`, the game-loop calibration all three sessions used |
| [`models/`](models) | Training and ONNX export script, and the manifest of the exported models (SHA-256, versions) |
| [`analysis/`](analysis) | Analysis, output-verification, and figure scripts |
| [`data/`](data) | Raw results, HWiNFO sensor logs, traces, and run metadata of every session ([data dictionary](data/README.md)) |
| [`results/tables/`](results/tables) | Tables produced by the analysis scripts; every number in the paper comes from them |
| [`paper/`](paper) | IEEE conference paper (LaTeX) and its figures |

## Sessions

| Session | Date | Tests | Conditions | Results |
|---|---|---|---|---|
| Main | 2026-10-11 | 2,160 | `BASELINE`, `P_SHARED`, `P_SMT`, `E_CORE` | [`data/main/`](data/main) |
| Control | 2026-10-12 | 1,080 | `P_SMT`, `SMT_IDLE` | [`data/control/`](data/control) |
| Validation | 2026-10-29 | 432 | the five above plus `E_CORE_SMT_IDLE`, `P_SHARED_SMT_IDLE`, `E_CORE_1LP`, `UNPINNED` | [`data/validation/`](data/validation) |

The main and control sessions cover 180 configurations (input widths 10, 100, and 1000; hidden widths 32
to 512; 1, 2, 4, or 8 layers; 50, 100, or 200 agents) in three trials. The validation session repeats 16 of
them (listed in [`validation_configs.txt`](harness/launchers/validation/validation_configs.txt)).

## Regenerate the tables and figures

The analysis needs Python 3.11 and the packages in [`requirements.txt`](requirements.txt); it reads the
gzipped logs directly. Run from the repository root:

```bash
pip install -r requirements.txt

python analysis/audit_analysis.py --main data/main/benchmark_steady_state.csv --main-hwinfo data/main/HWiN_sensor_log.CSV.gz --control data/control/benchmark_smt_control.csv --control-hwinfo data/control/HWiN_sensor_log_smt.CSV.gz --out out/tables

python analysis/doc_tables.py --main data/main/benchmark_steady_state.csv --main-hwinfo data/main/HWiN_sensor_log.CSV.gz --control data/control/benchmark_smt_control.csv --control-hwinfo data/control/HWiN_sensor_log_smt.CSV.gz --out out/tables/doc

python analysis/validation_analysis.py --validation data/validation/benchmark_validation.csv --batches data/validation/validation_trace_batches.csv.gz --frames data/validation/validation_trace_frames.csv.gz --hwinfo data/validation/hwinfo_validation.CSV.gz --main data/main/benchmark_steady_state.csv --main-hwinfo data/main/HWiN_sensor_log.CSV.gz --control data/control/benchmark_smt_control.csv --control-hwinfo data/control/HWiN_sensor_log_smt.CSV.gz --out out/tables/validation

python analysis/make_thesis_figures.py --tables out/tables --figs out/figures
```

The files in `out/tables` are byte-identical to [`results/tables`](results/tables), with two exceptions that
come from elsewhere:

- `cpp_output_check.csv` is produced by `verify_cpp_outputs.py` (below).
- `setup.csv` was compiled by hand from msinfo32, HWiNFO, and the run metadata.

Figs. 1 to 3 also regenerate byte-identical to [`paper/figures`](paper/figures). In the paper:

- **Table I:** `absolute.csv` (measured placement and control frame times), `effects_summary.csv` (paired changes, CIs, Wilcoxon tests, latency ratios), and `deadlines.csv` (configuration and deadline counts).
- **Figures:** `paper_fig1_placement`, `paper_fig2_game_cost`, and `paper_fig3_ai_latency` in `out/figures` are Figs. 1 to 3.
- **Validation (Section IV-F):** the numbers are in `validation/facts.csv`.

### Output check

The harness wrote every model's outputs to [`data/output_check/`](data/output_check), on LP 0 and on LP 12.
To compare them with a float64 PyTorch reference, first regenerate the models (next section), then run:

```bash
python analysis/verify_cpp_outputs.py --models models_latest --cpp data/output_check/verify_out_lp0 data/output_check/verify_out_lp12 --out out/cpp_output_check.csv
```

## Rerun the benchmark

You need Windows on an Intel hybrid CPU, Visual Studio 2022 (MSVC v143), and HWiNFO64.

1. **Models.** Run `python models/path_predicting_training.py --output-dir models_latest`. It exports 200
   models, about 2 GB, so they are not in the repository; the benchmark grid uses 60 of them. Each model is
   trained from its own fixed seed. [`models/manifest.csv`](models/manifest.csv) records the files that were
   timed: parameter counts, SHA-256 hashes, and library versions.
2. **Build.** Open `harness/LocalSequenceGameOptimizationCores.sln`, restore the NuGet packages
   (ONNX Runtime 1.27.1), and build Release x64. The executable is written to `harness/x64/Release`.
3. **Main and control sessions.** Copy into `harness/x64/Release`:
   - the batch files from `harness/launchers/release`;
   - the `models_latest` folder;
   - optionally `harness/config/calibrated_loops.txt`, to reuse the paper's calibration.

   Start HWiNFO64 CSV logging, then run `run_benchmark.bat` (main session, seed 1337). Afterwards run
   `run_smt_control.bat` (control session).
4. **Validation session.** Create `harness/x64/Validation` containing:
   - the executable and its ONNX Runtime DLLs;
   - the files from `harness/launchers/validation`;
   - `calibrated_loops.txt`.

   Run `run_validation.bat` (seed 4242). It expects the models in `harness/x64/Release/models_latest`.
5. **Output check.** Run `LocalSequenceGameOptimizationCores.exe --verify-outputs DIR --verify-lp N`. It
   writes every grid model's outputs to `DIR` and exits without timing anything.

`LocalSequenceGameOptimizationCores.exe --help` lists every option. The launchers stop Windows Update and
SysMain and check the power plan. They restore both afterwards, even if the benchmark fails.

## Test system

- **Laptop:** MSI Pulse GL66 12UEK with an Intel Core i9-12900H (6 P-cores with 12 LPs, 8 E-cores, 24-MiB last-level cache) and 32 GB of DDR4-3200.
- **Operating system:** Windows 11 Pro 25H2 (build 26200.9457), with virtualization-based security and hypervisor-enforced code integrity on.
- **Power:** on AC. MSI Center Extreme Performance mode, which sets PL1 = PL2 = 200 W. Windows Best performance mode on the Balanced plan.
- **Software:** MSVC 19.39 (C++20, /O2); ONNX Runtime 1.27.1 with the default CPU execution provider; PyTorch 2.7.1 and ONNX 1.19.1 (opset 17) for the models.
