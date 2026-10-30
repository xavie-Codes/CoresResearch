@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Hardware Benchmark Automation

:: =====================================================================================
:: run_benchmark.bat
:: Launches LocalSequenceGameOptimizationCores.exe in a controlled environment and ALWAYS
:: restores the power plan and the services it stopped, whatever the benchmark's exit code.
:: Extra arguments are passed through to the benchmark, e.g.:  run_benchmark.bat --max-tests 200
:: A log of every launch is appended to run_benchmark_log.txt.
:: =====================================================================================

:: 1. Check for Administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Please right-click this script and select "Run as Administrator".
    pause
    exit /b 1
)

:: Fix: Force the working directory to the folder containing this batch script
cd /d "%~dp0"

set "EXE=LocalSequenceGameOptimizationCores.exe"
set "LOG=run_benchmark_log.txt"
set "HIGH_PERF_GUID=8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
set "BENCH_EXIT=not run"

if not exist "%EXE%" (
    echo [ERROR] %EXE% not found in "%CD%".
    pause
    exit /b 1
)
if not exist "models_latest\" (
    echo [ERROR] models_latest folder not found in "%CD%". Copy it from the Python project first.
    pause
    exit /b 1
)

>>"%LOG%" echo.
>>"%LOG%" echo ==== run_benchmark.bat started %DATE% %TIME% ====

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

:: 3. Operator checklist
echo.
echo ===================== BEFORE YOU CONTINUE =====================
echo  1. MSI Center: select the SAME performance mode used in every previous
echo     session (e.g. Extreme Performance) and keep it fixed for the whole run.
echo  2. Windows Settings ^> System ^> Power: keep the same Power mode every session.
echo  3. Close background apps: browsers, launchers, cloud sync, RGB/overlay tools.
echo  4. Start HWiNFO64 sensor CSV logging now (package temp, package power, clocks).
echo  5. Leave the machine alone. Ctrl+C stops the benchmark cleanly; if cmd then
echo     asks "Terminate batch job (Y/N)?" answer N so the power plan and services
echo     are restored.
echo ================================================================
pause

:: 4. Power plan: remember the current scheme, then try High Performance
set "ORIG_SCHEME="
for /f "tokens=2 delims=:(" %%A in ('powercfg /getactivescheme') do for /f "tokens=1" %%B in ("%%A") do set "ORIG_SCHEME=%%B"
if not defined ORIG_SCHEME (
    echo [WARN] Could not read the active power scheme; it cannot be restored afterwards.
    >>"%LOG%" echo WARN: could not read the active power scheme
) else (
    echo [2/6] Current power scheme: %ORIG_SCHEME%
    >>"%LOG%" echo Original power scheme: %ORIG_SCHEME%
)

powercfg /list | findstr /i "%HIGH_PERF_GUID%" >nul
if errorlevel 1 (
    echo [WARN] The High Performance plan ^(%HIGH_PERF_GUID%^) does not exist on this
    echo        system ^(common on Modern Standby laptops^). Continuing on the current scheme.
    >>"%LOG%" echo WARN: High Performance plan not installed; staying on the current scheme
) else (
    powercfg /setactive %HIGH_PERF_GUID%
    if errorlevel 1 (
        echo [WARN] powercfg /setactive %HIGH_PERF_GUID% failed; continuing on the current scheme.
        >>"%LOG%" echo WARN: powercfg /setactive %HIGH_PERF_GUID% failed
    )
)
set "ACTIVE_SCHEME_LINE=unknown"
for /f "delims=" %%L in ('powercfg /getactivescheme') do set "ACTIVE_SCHEME_LINE=%%L"
echo [2/6] Active during benchmark: %ACTIVE_SCHEME_LINE%
>>"%LOG%" echo Active during benchmark: %ACTIVE_SCHEME_LINE%

:: 5. Background services: stop only those that were running, and remember which
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

:: 6. Run the benchmark (it sets HIGH_PRIORITY_CLASS itself). Arguments are passed through.
::    Full path: works even when NoDefaultCurrentDirectoryInExePath is set, and avoids PATH lookups.
echo [4/6] Launching %EXE% %*
echo ------------------------------------------------------
"%~dp0%EXE%" %*
set "BENCH_EXIT=%ERRORLEVEL%"
echo ------------------------------------------------------
>>"%LOG%" echo Benchmark exit code: %BENCH_EXIT%

:: 7. Restore -- always reached, whatever the benchmark's exit code
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
if defined ORIG_SCHEME (
    powercfg /setactive %ORIG_SCHEME%
    if errorlevel 1 (echo [WARN] could not restore power scheme %ORIG_SCHEME%) else (echo        power scheme restored to %ORIG_SCHEME%)
)
>>"%LOG%" echo Restored: services wuauserv=%WU_WAS_RUNNING% SysMain=%SYSMAIN_WAS_RUNNING%, scheme=%ORIG_SCHEME%

:: 8. Report the benchmark's exit code
echo [6/6] Benchmark exit code: %BENCH_EXIT%
if "%BENCH_EXIT%"=="0" (
    echo        Session finished normally.
) else if "%BENCH_EXIT%"=="130" (
    echo        Stopped by Ctrl+C. Run this script again to resume.
) else if "%BENCH_EXIT%"=="4" (
    echo        AC power was lost during the run. Run again on AC to resume.
) else if "%BENCH_EXIT%"=="2" (
    echo        Invalid command-line arguments.
) else if "%BENCH_EXIT%"=="1" (
    echo        Setup refused or failed ^(battery, topology or CSV header^). See the messages above.
) else (
    echo        [WARN] Benchmark failed or crashed. Completed rows are already in the CSV; run again to resume.
)
>>"%LOG%" echo ==== finished %DATE% %TIME% (exit %BENCH_EXIT%) ====

echo.
echo Automation script finished.
pause
endlocal & exit /b %BENCH_EXIT%
