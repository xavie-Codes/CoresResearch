# Data dictionary

All three sessions ran on the same laptop with the same game-loop calibration
([`harness/config/calibrated_loops.txt`](../harness/config/calibrated_loops.txt), loop count 338571). Times
in the results files are local time (UTC-05:00).

## Sessions and files

| Session | Run_ID | Time | Tests | Seed | Files |
|---|---|---|---|---|---|
| Main | `20261011-041929` | 2026-10-11 04:19 to 16:22 | 2,160 | 1337 | [`main/`](main) |
| Control | `20261012-005526` | 2026-10-12 00:55 to 04:44 | 1,080 | 1337 | [`control/`](control) |
| Validation | `20261029-191126` | 2026-10-29 19:11 to 21:51 | 432 | 4242 | [`validation/`](validation) |

- **Main:** `BASELINE`, `P_SHARED`, `P_SMT`, and `E_CORE` for 180 configurations in three trials.
- **Control:** `P_SMT` and `SMT_IDLE` for the same blocks in the same order. `SMT_IDLE` is the matched control for `P_SMT`.
- **Validation:** nine conditions for 16 configurations. Every window lasted at least 20 s, and per-batch and per-frame traces were written.
- **Shared logs:** [`run_metadata_main_control.txt`](run_metadata_main_control.txt) and [`run_benchmark_log_main_control.txt`](run_benchmark_log_main_control.txt) hold both the main and the control session, appended one session after the other.
- **Smoke test:** `run_metadata_main_control.txt` also records session `20261012-004529`, a 2-test smoke test that wrote `smoke.csv`. That file is not part of the data.

| File | Contents |
|---|---|
| `main/benchmark_steady_state.csv` | One row per test of the main session |
| `main/HWiN_sensor_log.CSV.gz` | HWiNFO64 sensor log during the main session |
| `control/benchmark_smt_control.csv` | One row per test of the control session |
| `control/HWiN_sensor_log_smt.CSV.gz` | HWiNFO64 sensor log during the control session |
| `validation/benchmark_validation.csv` | One row per test of the validation session (results columns plus the validation columns) |
| `validation/validation_trace_batches.csv.gz` | One row per measured AI batch |
| `validation/validation_trace_frames.csv.gz` | One row per measured game frame |
| `validation/hwinfo_validation.CSV.gz` | HWiNFO64 sensor log during the validation session (2-s interval) |
| `validation/run_metadata.txt`, `validation/run_validation_log.txt` | Validation session metadata and launcher log |
| `output_check/verify_out_lp0/`, `output_check/verify_out_lp12/` | Model outputs written by `--verify-outputs` on LP 0 and LP 12 |

### Run metadata

Each session block in a `run_metadata*.txt` file records:

- **System:** CPU topology and the LP masks, the Windows build, and the VBS/HVCI registry state.
- **Build:** the ONNX Runtime version and the compiler.
- **Power:** AC power, the power plan and overlay, and the services stopped.
- **Run settings:** the seed, the configuration grid, and the calibration.
- **End of session:** counts of written tests, drift flags, and truncated windows.

The validation session also records:

- `tsc_hz`, the time-stamp-counter rate behind `AI_OnCPU_Share`;
- the BIOS version;
- the operator's note: room temperature, MSI Center mode, and Cooler Boost.

## Results columns (one row per test)

| Column | Meaning |
|---|---|
| `Run_ID` | Session identifier (start time, `yyyymmdd-hhmmss`) |
| `Timestamp_Start`, `Timestamp_End` | Measurement window, ISO 8601 local time with milliseconds and UTC offset (use these to align the HWiNFO logs) |
| `Inputs`, `Hidden_Size`, `Layers`, `Agent_Density` | Model configuration: input width, hidden width, GRU layers, and agents per batch |
| `Condition` | Placement condition (see the harness header in `steady_state_compute_bound.cpp`) |
| `Trial` | Trial 1 to 3 |
| `Block_Pos` | 0-based position of this condition inside its block, which ran back to back in random order |
| `AI_LP_Mask` | Affinity mask given to the AI thread (hex; `0x1` = LP 0, `0x2` = LP 1, `0x7F000` = LP 12 to 18) |
| `Batch_Mode` | `batched` (all agents in one ONNX Runtime call) or `per_agent` (fallback; never used) |
| `Calibrated_Loop_Count` | Game-loop iterations per frame |
| `Pre_Test_Frame_ms` | Median of the 60-frame drift check before the window |
| `Drift_Ref_ms` | Reference for the drift check (16.0 ms, or 9.47 ms with the LP 1 worker parked) |
| `Drift_Flag` | 1 if the drift check was more than 15% from its reference |
| `Truncated` | 1 if the window hit the wall-time cap before the minimum frames and batches |
| `Measure_Window_s` | Length of the measurement window |
| `Game_Frames` | Game frames measured |
| `Game_Avg_ms`, `Game_Median_ms`, `Game_Min_ms`, `Game_99th_ms`, `Game_P999_ms`, `Game_Max_ms`, `Game_StdDev` | Game frame time statistics (percentiles are nearest-rank) |
| `Game_P1Low_FPS` | 1000 / `Game_99th_ms` |
| `AI_Batches` | Completed AI batches (updates) in the window |
| `AI_Avg_ms`, `AI_Median_ms`, `AI_99th_ms`, `AI_Max_ms` | Batch latency: from the AI thread's wake-up to the end of the batch |
| `AI_Agents_Per_Sec` | Agents per batch / mean batch latency |
| `AI_Missed_Deadlines` | Batches that ended after the end of their 50-ms update period |
| `AI_Duty_Cycle` | Share of the window spent inside batches (0 to 1) |
| `AI_Output_Checksum` | Checksum of the model outputs; identical for a configuration in every placement, trial, and session |
| `Telemetry_Samples` | PDH samples in the window (about one per 110 ms) |
| `Peak_/Mean_ACPI_ThermalZone_C` | Hottest ACPI thermal zone; not the CPU package temperature, and not used in the paper |
| `Peak_/Mean_PowerMeter_W` | ACPI power meter; reads 0 on AC power, so not used |
| `Peak_/Mean_RAPL_Package_W` | CPU package power from the RAPL energy meter |
| `Peak_/Mean_ProcPerf_P_Pct`, `Peak_/Mean_ProcPerf_E_Pct` | Windows `% Processor Performance`, averaged over the P-core or E-core LPs (above 100 when running above nominal frequency) |
| `Peak_PageFaults_per_s`, `Peak_FileCacheFaults_per_s` | System-wide page faults and file-system cache faults |

Columns added in the validation session:

| Column | Meaning |
|---|---|
| `Trace_ID` | `Run_ID-NNNN`; links the test to its rows in the trace files |
| `AI_Start_Delay_Median_ms`, `AI_Start_Delay_99th_ms`, `AI_Start_Delay_Max_ms` | Start delay: actual batch start minus its scheduled start (release) |
| `AI_OnCPU_Share` | Thread cycles / `tsc_hz` / batch latency, summed over the window: the share of batch latency the AI thread spent on a CPU |
| `AI_Start_LPs` | `LP:batches` pairs: where each batch started |
| `AI_Batches_LP_Changed` | Batches that ended on a different LP than they started on |

Unavailable PDH counters are written as -1.

## Trace columns (validation session)

All times are in ns from the start of the test's measurement window.

| File | Column | Meaning |
|---|---|---|
| batches | `Trace_ID`, `Batch` | Test and 0-based batch index |
| batches | `Release_ns` | Scheduled start: the start of the update period. After a missed deadline, the next period starts when that batch ends |
| batches | `Start_ns`, `End_ns` | Actual start and end of the batch |
| batches | `Missed` | 1 if `End_ns` is after `Release_ns` + 50 ms |
| batches | `LP_Start`, `LP_End` | LP the AI thread ran on at the start and at the end of the batch |
| batches | `Thread_Cycles` | `QueryThreadCycleTime` cycles the AI thread used during the batch |
| frames | `Trace_ID`, `Frame`, `Start_ns`, `End_ns` | Test, 0-based frame index, frame start and end |

## HWiNFO64 logs

These are HWiNFO64's own CSV format: Latin-1 encoded, with `Date` as `dd.mm.yyyy` and `Time` as `hh:mm:ss.fff`.
The last two rows repeat the header, so `analysis/audit_analysis.py:load_hwinfo` skips them.

The main session logged every 2 s, but HWiNFO refreshed its per-core sensors and package power only every 80 to
110 s (median). The control and validation sessions refreshed every 2 s, and the validation log includes no
fan sensors.

## Output check

Each `verify_out_*` folder holds `verify_info.txt` (run information), `verify_manifest.csv`, and one raw
little-endian float32 file per model, agent count, and input set. The input sets are the fixed benchmark
inputs (`bench`) and three seeded uniform inputs.

The manifest columns are:

- `inputs`, `hidden`, `layers`, `agents`: the configuration;
- `input_set`, `seed`, `lo`, `hi`: the input set, its seed, and its range;
- `input_bits_sum`: a checksum of the input bits;
- `outputs`: the number of output values;
- `all_finite`, `repeat_identical`: the finiteness and repeatability checks;
- `file`: the output file.

`analysis/verify_cpp_outputs.py` compares these files with a float64 PyTorch forward pass built from each
model's ONNX weights.
