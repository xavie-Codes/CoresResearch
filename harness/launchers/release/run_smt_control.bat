@echo off
:: =====================================================================================
:: run_smt_control.bat
:: Supplementary session for the paper: P_SMT and its matched control SMT_IDLE (the game
:: LP's SMT sibling parked and left idle, no AI), run back to back in every block.
::
:: It reuses calibrated_loops.txt and models_latest from the main session, so the frame
:: workload is identical, and writes to its own CSV (benchmark_smt_control.csv) so the
:: main session's P_SMT rows are not treated as already done.
::
:: Put this file next to run_benchmark.bat and LocalSequenceGameOptimizationCores.exe
:: (x64\Release), then right-click it and choose "Run as administrator".
:: run_benchmark.bat does the AC check, power plan, services, logging and restore steps.
:: Run it again after an interruption to resume.
:: =====================================================================================
cd /d "%~dp0"
if not exist "%~dp0run_benchmark.bat" (
    echo [ERROR] run_benchmark.bat not found next to this file.
    pause
    exit /b 1
)
if not exist "%~dp0calibrated_loops.txt" (
    echo [ERROR] calibrated_loops.txt not found. Copy it from the main session so the
    echo         frame workload matches; without it the benchmark recalibrates.
    pause
    exit /b 1
)
echo Supplementary session: conditions P_SMT and SMT_IDLE, results in benchmark_smt_control.csv
echo Optional: set HWiNFO64 to High priority (Task Manager, Details tab) so its per-core
echo sensors keep updating while every P-core is busy.
echo.
call "%~dp0run_benchmark.bat" --conditions "P_SMT,SMT_IDLE" --csv benchmark_smt_control.csv
exit /b %ERRORLEVEL%
