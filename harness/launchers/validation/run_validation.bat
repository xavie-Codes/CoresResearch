@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Validation Session

:: =====================================================================================
:: run_validation.bat
:: Focused validation session for the paper's review (about 3 hours):
::   16 configurations (validation_configs.txt) x 9 conditions x 3 trials = 432 tests.
::   Conditions: the paper's five (BASELINE, P_SHARED, P_SMT, E_CORE, SMT_IDLE), plus
::   E_CORE_SMT_IDLE and P_SHARED_SMT_IDLE (the same background work as P_SMT),
::   E_CORE_1LP (one pinned E-core LP) and UNPINNED (Windows chooses the AI thread's LP).
::   Every measurement window lasts at least 20 s, and every AI batch and game frame is
::   traced (release, start and end time, LP, thread cycles).
:: It reuses the main session's calibration (calibrated_loops.txt) and models
:: (..\Release\models_latest), and stops and restores the same services as
:: run_benchmark.bat. Right-click it and choose "Run as administrator".
:: Run it again after an interruption to resume.
:: Results in this folder: benchmark_validation.csv, validation_trace_batches.csv,
:: validation_trace_frames.csv, run_metadata.txt and run_validation_log.txt.
:: =====================================================================================

:: 1. Check for Administrator privileges (needed to stop and restart services)
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Please right-click this script and select "Run as administrator".
    pause
    exit /b 1
)

cd /d "%~dp0"

set "EXE=LocalSequenceGameOptimizationCores.exe"
set "MODELS=..\Release\models_latest"
set "LOG=run_validation_log.txt"
set "BENCH_EXIT=not run"

if not exist "%EXE%" (
    echo [ERROR] %EXE% not found in "%CD%".
    pause
    exit /b 1
)
if not exist "%MODELS%\" (
    echo [ERROR] %MODELS% not found. This folder must sit next to x64\Release.
    pause
    exit /b 1
)
if not exist "validation_configs.txt" (
    echo [ERROR] validation_configs.txt not found in "%CD%".
    pause
    exit /b 1
)
if not exist "calibrated_loops.txt" copy /y "..\Release\calibrated_loops.txt" "calibrated_loops.txt" >nul
findstr /b /c:"loop_count=338571" "calibrated_loops.txt" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] calibrated_loops.txt is missing or is not the main session's calibration
    echo         ^(loop_count=338571^). Copy it from ..\Release so the game workload matches.
    pause
    exit /b 1
)

>>"%LOG%" echo.
>>"%LOG%" echo ==== run_validation.bat started %DATE% %TIME% ====

:: 2. AC power check (GetSystemPowerStatus via .NET)
set "PWR_LINE=Unknown"
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus"`) do set "PWR_LINE=%%P"
echo [1/6] AC power status: %PWR_LINE%
>>"%LOG%" echo AC power: %PWR_LINE%
if /i not "%PWR_LINE%"=="Online" (
    echo [ERROR] Not on AC power. Plug in the charger and run this script again.
    >>"%LOG%" echo ABORT: not on AC power
    pause
    exit /b 1
)

:: 3. Power plan: the main session ran on Balanced with the Best performance power mode.
::    Switch to Balanced for the run (the original plan is restored at the end) and check
::    the power mode, which Windows stores as an overlay on the Balanced plan.
set "BALANCED_GUID=381b4222-f694-41f0-9685-ff5bb260df2e"
set "BEST_PERF_OVERLAY=ded574b5-45a0-4f42-8737-46345c09c238"
set "ORIG_SCHEME="
for /f "tokens=2 delims=:(" %%A in ('powercfg /getactivescheme') do for /f "tokens=1" %%B in ("%%A") do set "ORIG_SCHEME=%%B"
>>"%LOG%" echo Original power scheme: %ORIG_SCHEME%
if /i not "%ORIG_SCHEME%"=="%BALANCED_GUID%" (
    powercfg /setactive %BALANCED_GUID%
    if errorlevel 1 (
        echo [ERROR] Could not switch to the Balanced power plan.
        >>"%LOG%" echo ABORT: could not switch to Balanced
        pause
        exit /b 1
    )
    echo [2/6] Switched the power plan from %ORIG_SCHEME% to Balanced; it is restored at the end.
) else (
    echo [2/6] Power plan: Balanced
)
call :read_overlay
if /i not "!OVERLAY!"=="%BEST_PERF_OVERLAY%" (
    echo.
    echo [ACTION] Open Settings, System, Power ^& battery and set Power mode to
    echo          "Best performance", as in the main session. Then press a key here.
    pause
    call :read_overlay
)
if /i not "!OVERLAY!"=="%BEST_PERF_OVERLAY%" (
    echo [ERROR] Power mode is still not Best performance ^(overlay !OVERLAY!^).
    >>"%LOG%" echo ABORT: power mode overlay !OVERLAY!
    if /i not "%ORIG_SCHEME%"=="%BALANCED_GUID%" if defined ORIG_SCHEME powercfg /setactive %ORIG_SCHEME%
    pause
    exit /b 1
)
echo        Power mode: Best performance
>>"%LOG%" echo Power plan Balanced, power mode overlay !OVERLAY!

:: 4. Operator checklist
echo.
echo ===================== BEFORE YOU CONTINUE =====================
echo  1. Same setup as the main session: MSI Center Extreme Performance,
echo     airplane mode on, on AC power.
echo  2. Close every other application (browsers, launchers, cloud sync,
echo     RGB and overlay tools, Claude). Restart first if Windows is updating.
echo  3. Start HWiNFO64 sensor CSV logging at a 2000-ms interval, including the
echo     fan sensors, then set HWiNFO64 to High priority (Task Manager, Details).
echo  4. Note the room temperature; you are asked for it next.
echo  5. The run takes about 3 hours. Leave the laptop alone on a hard, flat
echo     surface. Ctrl+C stops it cleanly; if cmd asks "Terminate batch job
echo     (Y/N)?" answer N so the services are restored. Run again to resume.
echo ================================================================
pause

:: 5. Session conditions for the metadata (plain words and numbers only)
echo.
echo Describe this session's conditions. Use plain words and numbers only.
set "AMBIENT=not given"
set "MSIMODE=not given"
set "COOLER=not given"
set "SURFACE=not given"
set /p "AMBIENT=  Room temperature in C (e.g. 23): "
set /p "MSIMODE=  MSI Center mode (e.g. Extreme Performance): "
set /p "COOLER=  Cooler Boost (on or off): "
set /p "SURFACE=  Surface (e.g. desk, no cooling pad): "
set "NOTE=room_C=!AMBIENT!; msi_mode=!MSIMODE!; cooler_boost=!COOLER!; surface=!SURFACE!"
>>"%LOG%" echo Note: !NOTE!

:: 6. Background services: stop only those that were running, and remember which
set "WU_WAS_RUNNING=0"
set "SYSMAIN_WAS_RUNNING=0"
sc query wuauserv | find "RUNNING" >nul && set "WU_WAS_RUNNING=1"
sc query SysMain | find "RUNNING" >nul && set "SYSMAIN_WAS_RUNNING=1"
echo [3/6] Services before run: wuauserv running=%WU_WAS_RUNNING%, SysMain running=%SYSMAIN_WAS_RUNNING%
>>"%LOG%" echo Services before run: wuauserv=%WU_WAS_RUNNING% SysMain=%SYSMAIN_WAS_RUNNING%
if "%WU_WAS_RUNNING%"=="1" (
    net stop wuauserv /y >nul 2>&1
    if errorlevel 1 (echo [WARN] could not stop wuauserv) else (echo        stopped wuauserv)
)
if "%SYSMAIN_WAS_RUNNING%"=="1" (
    net stop SysMain /y >nul 2>&1
    if errorlevel 1 (echo [WARN] could not stop SysMain) else (echo        stopped SysMain)
)

:: 7. Run the validation session (the benchmark sets HIGH_PRIORITY_CLASS itself)
echo [4/6] Launching %EXE% (validation session)
echo ------------------------------------------------------
"%~dp0%EXE%" --conditions "BASELINE,SMT_IDLE,P_SMT,E_CORE,P_SHARED,E_CORE_SMT_IDLE,P_SHARED_SMT_IDLE,E_CORE_1LP,UNPINNED" --configs validation_configs.txt --min-window-seconds 20 --trace validation_trace --csv benchmark_validation.csv --model-dir "%MODELS%" --seed 4242 --note "!NOTE!"
set "BENCH_EXIT=%ERRORLEVEL%"
echo ------------------------------------------------------
>>"%LOG%" echo Benchmark exit code: %BENCH_EXIT%

:: 8. Restore -- always reached, whatever the benchmark's exit code
echo [5/6] Restoring services and power plan...
if "%WU_WAS_RUNNING%"=="1" (
    sc query wuauserv | find "RUNNING" >nul
    if errorlevel 1 (
        net start wuauserv >nul 2>&1
        if errorlevel 1 (echo [WARN] could not restart wuauserv) else (echo        restarted wuauserv)
    ) else (
        echo        wuauserv is already running ^(trigger-started^)
    )
)
if "%SYSMAIN_WAS_RUNNING%"=="1" (
    sc query SysMain | find "RUNNING" >nul
    if errorlevel 1 (
        net start SysMain >nul 2>&1
        if errorlevel 1 (echo [WARN] could not restart SysMain) else (echo        restarted SysMain)
    ) else (
        echo        SysMain is already running
    )
)
if defined ORIG_SCHEME if /i not "%ORIG_SCHEME%"=="%BALANCED_GUID%" (
    powercfg /setactive %ORIG_SCHEME%
    if errorlevel 1 (echo [WARN] could not restore power scheme %ORIG_SCHEME%) else (echo        power scheme restored to %ORIG_SCHEME%)
)
>>"%LOG%" echo Restored: services wuauserv=%WU_WAS_RUNNING% SysMain=%SYSMAIN_WAS_RUNNING%, scheme=%ORIG_SCHEME%

:: 9. Report the benchmark's exit code
echo [6/6] Benchmark exit code: %BENCH_EXIT%
if "%BENCH_EXIT%"=="0" (
    echo        Session finished normally.
) else if "%BENCH_EXIT%"=="130" (
    echo        Stopped by Ctrl+C. Run this script again to resume.
) else if "%BENCH_EXIT%"=="4" (
    echo        AC power was lost during the run. Run again on AC to resume.
) else if "%BENCH_EXIT%"=="2" (
    echo        Invalid command-line arguments or configuration list.
) else if "%BENCH_EXIT%"=="1" (
    echo        Setup refused or failed ^(battery, topology, CSV header or trace files^). See the messages above.
) else (
    echo        [WARN] Benchmark failed or crashed. Completed rows are already in the CSV; run again to resume.
)
>>"%LOG%" echo ==== finished %DATE% %TIME% (exit %BENCH_EXIT%) ====

echo.
echo Validation script finished.
pause
endlocal & exit /b %BENCH_EXIT%

:: Reads the Windows power mode overlay of the AC power plan into OVERLAY.
:read_overlay
set "OVERLAY=not set"
for /f "tokens=1,3" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes" /v ActiveOverlayAcPowerScheme 2^>nul') do if /i "%%A"=="ActiveOverlayAcPowerScheme" set "OVERLAY=%%B"
exit /b 0
