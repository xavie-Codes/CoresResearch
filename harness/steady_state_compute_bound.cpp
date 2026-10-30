// =====================================================================================================
// steady_state_compute_bound.cpp
//
// Hybrid-CPU placement benchmark: ONNX GRU path-prediction inference ("enemy AI") co-running with a
// mock game loop on an Intel hybrid CPU (developed on a Core i9-12900H: 6 P-cores / 12 threads +
// 8 E-cores = 20 logical processors in a single processor group).
//
// Session flow
//   1. Process setup: HIGH_PRIORITY_CLASS, power throttling (EcoQoS execution speed + timer-resolution
//      throttling) disabled, AC-power check (refuses battery unless --allow-battery).
//   2. Topology: LP0 = lowest logical processor (LP) of the first physical P-core (max EfficiencyClass);
//      LP_sib = its SMT sibling. Background "worker" threads are started ONCE, one per P-core LP except
//      LP0, and run continuously until program exit (persistent steady-state load). Individual workers
//      are parked/resumed when a test condition needs their LP.
//   3. Soak (default 180 s) under full load so the laptop leaves its PL2 boost window, then calibrate
//      the physics loop count so one frame takes 16.0 ms (median) on LP0 under load. The loop count is
//      persisted with its calibration timestamp in calibrated_loops.txt and reused by resumed sessions.
//   4. Test plan: for every (inputs, hidden, layers, density, trial) block, all placement conditions run
//      back-to-back in random order; the block order is also shuffled (fixed, logged seed). Resumable.
//   5. Per test: apply the condition's worker set -> 60-frame pre-test drift check on LP0 -> launch game
//      (+ AI) -> model load and warm-up (not measured; the game thread keeps LP0 busy) -> synchronized
//      measurement window -> one CSV row. Workers are never stopped between tests.
//
// Placement conditions (explicit, single-LP)
//   BASELINE : no AI thread. Game on LP0; workers on every P LP except LP0.
//   P_SHARED : AI pinned to LP0 (time-slices with the game thread). Workers on every P LP except LP0.
//   P_SMT    : AI pinned to LP_sib, whose worker is parked for this test. Workers on all other P LPs.
//   SMT_IDLE : no AI thread; LP_sib's worker is parked and LP_sib is left idle. Workers on all other P LPs.
//              The matched control for P_SMT (same worker set, no AI). Not in the default list; select it
//              with --conditions, e.g. --conditions "P_SMT,SMT_IDLE" --csv benchmark_smt_control.csv
//   E_CORE   : AI pinned to the E-core mask (minus one "housekeeping" E-core LP reserved for the
//              telemetry/orchestrator threads when >= 2 E-core LPs exist; see AI_LP_Mask).
//              Workers on every P LP except LP0.
//
// Validation conditions (not in the default list; select them with --conditions)
//   E_CORE_SMT_IDLE   : E_CORE with LP_sib's worker parked, as in P_SMT. Control: SMT_IDLE.
//   P_SHARED_SMT_IDLE : P_SHARED with LP_sib's worker parked, as in P_SMT. Control: SMT_IDLE.
//                       With P_SMT, the three placements then run with the same background work.
//   E_CORE_1LP        : AI pinned to one E-core LP (the lowest LP of the E_CORE mask). Control: BASELINE.
//   UNPINNED          : the AI thread makes no affinity call and keeps the process affinity, so Windows
//                       chooses its LP. Workers on every P LP except LP0. Control: BASELINE.
//
// Traces (--trace PREFIX): <PREFIX>_batches.csv has one row per measured AI batch (release, start and end
// time, deadline miss, LP at batch start and end, thread cycles) and <PREFIX>_frames.csv one row per measured
// game frame (start and end time). Times are in ns from the start of the measurement window; Trace_ID links
// the rows to the test's row in the results CSV.
//
// Telemetry (PDH) -- coarse OS-level signals; column names state exactly what is read:
//   *_ACPI_ThermalZone_C : hottest ACPI thermal zone (\_TZ.*). This is NOT the CPU package temperature.
//   *_PowerMeter_W       : ACPI "Power Meter" (battery/system meter; reads 0 on AC on many laptops).
//   *_RAPL_Package_W     : "Energy Meter(RAPL_Package0_PKG)\Power" (CPU package power via RAPL).
//   *_ProcPerf_{P,E}_Pct : "% Processor Performance" averaged over the P-core / E-core LP instances
//                          (exceeds 100 when running above nominal frequency).
//   Peak_PageFaults_per_s       : \Memory\Page Faults/sec (system-wide, all processes).
//   Peak_FileCacheFaults_per_s  : \Memory\Cache Faults/sec = FILE-SYSTEM cache faults (system-wide), not CPU cache.
//   Unavailable counters are written as -1.
//
//   For package temperature, package power and per-core clocks, run HWiNFO64 sensor CSV logging in
//   parallel for the whole session and align it to this benchmark's Timestamp_Start / Timestamp_End
//   columns (ISO 8601 local time with milliseconds and UTC offset).
//
// Outputs (current working directory): benchmark_steady_state.csv, calibrated_loops.txt,
// run_metadata.txt (appended once per session).
// =====================================================================================================

#define NOMINMAX
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <powrprof.h>
#include <timeapi.h>
#include <intrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#pragma comment(lib, "pdh.lib")
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "powrprof.lib")
#pragma comment(lib, "advapi32.lib")

// Older SDK headers may not define these (values from the Windows 11 SDK).
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
#ifndef PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION
#define PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION 0x4
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock; // QueryPerformanceCounter on MSVC

// -----------------------------------------------------------------------------------------------------
// Fixed experiment constants
// -----------------------------------------------------------------------------------------------------
constexpr double    TARGET_FRAME_MS = 16.0;          // calibrated physics frame time on LP0 under load
constexpr double    CALIBRATION_TOLERANCE_MS = 0.16; // 1% of target
constexpr int       CALIBRATION_MAX_ITERS = 30;
constexpr int       CALIBRATION_FRAMES = 15;         // frames per calibration iteration (median)
constexpr int       VERIFY_FRAMES = 60;
constexpr double    DRIFT_TOLERANCE = 0.15;          // Drift_Flag if |pre-test - reference| / reference > 15%
constexpr int       DRIFT_FRAMES = 60;
constexpr int       DRIFT_WARMUP_FRAMES = 5;
constexpr int       GAME_WARMUP_FRAMES = 5;
constexpr int       AI_WARMUP_BATCHES = 5;
constexpr int       SEQ_LEN = 10;                    // GRU sequence length (must match the Python export)
constexpr long long DEFAULT_LOOP_COUNT = 120000;
constexpr int       POLL_MS = 5;
constexpr int       PARK_TIMEOUT_MS = 5000;
constexpr double    SOAK_REPORT_WINDOW_S = 10.0;

// 1.5 MB per-thread working set (393,216 floats) touched with a 64-byte (16-float) stride.
constexpr size_t PHYSICS_BUFFER_ELEMENTS = 393216;
constexpr size_t PHYSICS_STRIDE = 16;

// -----------------------------------------------------------------------------------------------------
// Run configuration (defaults; overridable from the command line, see printUsage)
// -----------------------------------------------------------------------------------------------------
struct BenchConfig {
    bool allowBattery = false;
    bool recalibrate = false;
    int soakSeconds = 180;
    int trials = 3;
    double tickMs = 50.0;            // AI tick period (50 ms = 20 Hz)
    int minGameFrames = 1000;
    int minAIBatches = 50;
    int maxAIBatches = 2000;
    double maxTestSeconds = 300.0;   // wall-time cap per measurement window -> Truncated flag
    uint32_t seed = 1337;
    int maxTests = -1;               // stop the session after N written tests (-1 = no limit)
    int telemetryIntervalMs = 100;
    std::string csvPath = "benchmark_steady_state.csv";
    std::string calibrationPath = "calibrated_loops.txt";
    std::string metadataPath = "run_metadata.txt";
    std::string modelDir = "models_latest";
    std::string conditions;          // --conditions list; empty = the default list in main()
    std::string verifyDir;           // --verify-outputs: write every grid model's outputs here and exit
    int verifyLP = -1;               // --verify-lp: LP the verification runs on (-1 = not pinned)
    std::string configsPath;         // --configs: run only the configurations listed in this file
    double minWindowSeconds = 0.0;   // --min-window-seconds: the window also lasts at least this long
    std::string tracePrefix;         // --trace: write per-batch and per-frame traces (empty = no traces)
    std::string note;                // --note: free-text operator note written to the metadata
};

// -----------------------------------------------------------------------------------------------------
// Placement conditions
// -----------------------------------------------------------------------------------------------------
enum class Condition { BASELINE, P_SHARED, P_SMT, E_CORE, SMT_IDLE, E_CORE_SMT_IDLE, P_SHARED_SMT_IDLE, E_CORE_1LP, UNPINNED };

constexpr Condition ALL_CONDITIONS[] = { Condition::BASELINE, Condition::P_SHARED, Condition::P_SMT, Condition::E_CORE,
    Condition::SMT_IDLE, Condition::E_CORE_SMT_IDLE, Condition::P_SHARED_SMT_IDLE, Condition::E_CORE_1LP, Condition::UNPINNED };

const char* conditionName(Condition c) {
    switch (c) {
    case Condition::BASELINE:          return "BASELINE";
    case Condition::P_SHARED:          return "P_SHARED";
    case Condition::P_SMT:             return "P_SMT";
    case Condition::E_CORE:            return "E_CORE";
    case Condition::SMT_IDLE:          return "SMT_IDLE";
    case Condition::E_CORE_SMT_IDLE:   return "E_CORE_SMT_IDLE";
    case Condition::P_SHARED_SMT_IDLE: return "P_SHARED_SMT_IDLE";
    case Condition::E_CORE_1LP:        return "E_CORE_1LP";
    case Condition::UNPINNED:          return "UNPINNED";
    }
    return "UNKNOWN";
}

bool parseCondition(const std::string& s, Condition& out) {
    for (Condition c : ALL_CONDITIONS) {
        if (s == conditionName(c)) { out = c; return true; }
    }
    return false;
}

// BASELINE and SMT_IDLE are the no-AI controls; every other condition launches an AI thread.
bool hasAIThread(Condition c) { return c != Condition::BASELINE && c != Condition::SMT_IDLE; }

// P_SMT, its control SMT_IDLE and the two same-work placements park the worker on the game LP's SMT sibling.
bool parksSibling(Condition c) {
    return c == Condition::P_SMT || c == Condition::SMT_IDLE || c == Condition::E_CORE_SMT_IDLE || c == Condition::P_SHARED_SMT_IDLE;
}

// Conditions whose AI thread runs on the E-cores (removed from the plan on a CPU that is not hybrid).
bool needsECores(Condition c) { return c == Condition::E_CORE || c == Condition::E_CORE_SMT_IDLE || c == Condition::E_CORE_1LP; }

// -----------------------------------------------------------------------------------------------------
// Topology
// -----------------------------------------------------------------------------------------------------
struct Topology {
    bool hybrid = false;
    int logicalCount = 0;
    int pPhysical = 0;
    int ePhysical = 0;
    int otherPhysical = 0;         // cores with an EfficiencyClass between min and max (ignored)
    DWORD_PTR processMask = 0;
    DWORD_PTR pCoreMask = 0;       // all LPs of max-EfficiencyClass cores
    DWORD_PTR eCoreMask = 0;       // all LPs of min-EfficiencyClass cores (0 if not hybrid)
    int gameLP = -1;               // LP0: lowest LP of the first physical P-core
    int siblingLP = -1;            // SMT sibling of gameLP (-1 if that core has no SMT)
    DWORD_PTR workerMask = 0;      // pCoreMask without gameLP
    int housekeepingLP = -1;       // E-core LP for telemetry + orchestrator (-1 if none)
    DWORD_PTR eCoreAIMask = 0;     // AI mask for E_CORE
};
Topology g_topo;

inline DWORD_PTR lpBit(int lp) { return static_cast<DWORD_PTR>(1) << lp; }
inline int lowestLP(DWORD_PTR m) { return m ? std::countr_zero(static_cast<unsigned long long>(m)) : -1; }
inline int highestLP(DWORD_PTR m) { return m ? 63 - std::countl_zero(static_cast<unsigned long long>(m)) : -1; }
inline int lpCount(DWORD_PTR m) { return std::popcount(static_cast<unsigned long long>(m)); }

std::string hexMask(DWORD_PTR m) {
    char b[32];
    snprintf(b, sizeof(b), "0x%llX", static_cast<unsigned long long>(m));
    return b;
}

std::string lpList(DWORD_PTR m) {
    std::string out;
    int lp = 0;
    while (lp < 64) {
        if (!(m & lpBit(lp))) { ++lp; continue; }
        int end = lp;
        while (end + 1 < 64 && (m & lpBit(end + 1))) ++end;
        if (!out.empty()) out += ",";
        out += (end == lp) ? std::to_string(lp) : std::to_string(lp) + "-" + std::to_string(end);
        lp = end + 1;
    }
    return out.empty() ? "none" : out;
}

// -----------------------------------------------------------------------------------------------------
// Globals
// -----------------------------------------------------------------------------------------------------
std::atomic<long long> g_loopCount{ DEFAULT_LOOP_COUNT };
std::atomic<bool> g_shutdown{ false };
std::atomic<bool> g_abortRequested{ false };
bool g_highResTimerAvailable = false;

// -----------------------------------------------------------------------------------------------------
// Small utilities
// -----------------------------------------------------------------------------------------------------
std::string narrow(const std::wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

std::string pdhStatusHex(PDH_STATUS s) {
    char b[32];
    snprintf(b, sizeof(b), "0x%08lX", static_cast<unsigned long>(s));
    return b;
}

double msBetween(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double secondsSince(Clock::time_point a) {
    return std::chrono::duration<double>(Clock::now() - a).count();
}

std::string formatDuration(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0) return "n/a";
    long long s = static_cast<long long>(seconds + 0.5);
    long long d = s / 86400; s %= 86400;
    long long h = s / 3600;  s %= 3600;
    long long m = s / 60;    s %= 60;
    char b[64];
    if (d > 0) snprintf(b, sizeof(b), "%lldd %02lldh %02lldm", d, h, m);
    else snprintf(b, sizeof(b), "%02lldh %02lldm %02llds", h, m, s);
    return b;
}

FILETIME nowUtcFileTime() {
    FILETIME ft{};
    GetSystemTimePreciseAsFileTime(&ft);
    return ft;
}

long long fileTimeToI64(const FILETIME& f) {
    return (static_cast<long long>(f.dwHighDateTime) << 32) | static_cast<long long>(f.dwLowDateTime);
}

FILETIME fileTimeAddMs(const FILETIME& f, double ms) {
    const long long v = fileTimeToI64(f) + static_cast<long long>(std::llround(ms * 10000.0)); // 100 ns units
    FILETIME out{};
    out.dwLowDateTime = static_cast<DWORD>(v & 0xFFFFFFFFLL);
    out.dwHighDateTime = static_cast<DWORD>((v >> 32) & 0xFFFFFFFFLL);
    return out;
}

// ISO 8601 local time with milliseconds and UTC offset, e.g. 2026-10-10T23:15:00.123+02:00
std::string isoLocal(const FILETIME& ftUtc) {
    SYSTEMTIME stUtc{}, stLocal{};
    if (!FileTimeToSystemTime(&ftUtc, &stUtc) || !SystemTimeToTzSpecificLocalTime(nullptr, &stUtc, &stLocal)) return "";
    FILETIME ftUtcMs{}, ftLocalMs{};
    SystemTimeToFileTime(&stUtc, &ftUtcMs);   // both truncated to ms, so the difference is the exact offset
    SystemTimeToFileTime(&stLocal, &ftLocalMs);
    const long long offsetMin = (fileTimeToI64(ftLocalMs) - fileTimeToI64(ftUtcMs)) / 600000000LL;
    const long long absMin = offsetMin < 0 ? -offsetMin : offsetMin;
    auto u = [](WORD w) { return static_cast<unsigned>(w); };
    char b[64];
    snprintf(b, sizeof(b), "%04u-%02u-%02uT%02u:%02u:%02u.%03u%c%02lld:%02lld",
        u(stLocal.wYear), u(stLocal.wMonth), u(stLocal.wDay), u(stLocal.wHour), u(stLocal.wMinute),
        u(stLocal.wSecond), u(stLocal.wMilliseconds), offsetMin < 0 ? '-' : '+', absMin / 60, absMin % 60);
    return b;
}

std::string isoLocalNow() { return isoLocal(nowUtcFileTime()); }

std::string makeRunId() {
    SYSTEMTIME st{};
    GetLocalTime(&st);
    char b[32];
    snprintf(b, sizeof(b), "%04u%02u%02u-%02u%02u%02u", static_cast<unsigned>(st.wYear),
        static_cast<unsigned>(st.wMonth), static_cast<unsigned>(st.wDay), static_cast<unsigned>(st.wHour),
        static_cast<unsigned>(st.wMinute), static_cast<unsigned>(st.wSecond));
    return b;
}

// Aborts the whole process with a clear message. Used for unrecoverable errors in any thread (e.g. an
// affinity call failing): worker / ONNX Runtime threads cannot be joined from an arbitrary thread, so the
// process is terminated immediately. Every completed CSV row has already been flushed.
[[noreturn]] void fatal(const std::string& msg, int exitCode = 3) {
    std::cout.flush();
    std::cerr << "\n[FATAL] " << msg << std::endl;
    TerminateProcess(GetCurrentProcess(), static_cast<UINT>(exitCode));
    std::abort(); // not reached
}

// Pins the calling thread to 'mask', verifies the result with GetThreadGroupAffinity and returns the
// previous mask. Any failure aborts the program with a clear message.
DWORD_PTR pinCurrentThreadOrDie(DWORD_PTR mask, const std::string& who) {
    if (mask == 0) fatal("Affinity for " + who + ": empty mask");
    if ((mask & ~g_topo.processMask) != 0) {
        fatal("Affinity for " + who + ": mask " + hexMask(mask) + " is not a subset of the process affinity mask " +
            hexMask(g_topo.processMask));
    }
    const DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), mask);
    if (prev == 0) {
        fatal("SetThreadAffinityMask(" + hexMask(mask) + ") failed for " + who + ", GetLastError=" +
            std::to_string(GetLastError()));
    }
    GROUP_AFFINITY ga{};
    if (!GetThreadGroupAffinity(GetCurrentThread(), &ga)) {
        fatal("GetThreadGroupAffinity failed for " + who + ", GetLastError=" + std::to_string(GetLastError()));
    }
    if (ga.Group != 0 || ga.Mask != static_cast<KAFFINITY>(mask)) {
        fatal("Affinity verification failed for " + who + ": requested " + hexMask(mask) + ", got group " +
            std::to_string(ga.Group) + " mask " + hexMask(static_cast<DWORD_PTR>(ga.Mask)));
    }
    return prev;
}

bool waitUntilTrue(const std::function<bool()>& pred, int timeoutMs) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
    while (!pred()) {
        if (Clock::now() >= deadline) return false;
        Sleep(1);
    }
    return true;
}

// Implementation-independent unbiased integer in [0, n) from raw mt19937 output (the mt19937 sequence is
// fully specified by the C++ standard; std::shuffle / uniform_int_distribution are not).
uint32_t uniformBelow(std::mt19937& rng, uint32_t n) {
    const uint64_t range = 1ULL << 32;
    const uint64_t limit = range - (range % n);
    uint64_t x;
    do { x = static_cast<uint64_t>(rng()); } while (x >= limit);
    return static_cast<uint32_t>(x % n);
}

template <class T>
void deterministicShuffle(std::vector<T>& v, std::mt19937& rng) {
    for (size_t i = v.size(); i > 1; --i) {
        const size_t j = uniformBelow(rng, static_cast<uint32_t>(i));
        std::swap(v[i - 1], v[j]);
    }
}

// -----------------------------------------------------------------------------------------------------
// Mock physics frame
// -----------------------------------------------------------------------------------------------------
// The loop is bound by transcendental math (sin/cos/tan/sqrt every iteration); the strided touch of the
// 1.5 MB buffer adds only light cache pressure -- this is not a memory-bandwidth kernel. The volatile
// accumulator and the buffer write-back keep the compiler from eliding any iteration.
void executePhysicsMath(std::vector<float>& stateBuffer, long long loopCount) {
    volatile double sum = 0.0;
    size_t idx = 0;

    for (long long j = 0; j < loopCount; ++j) {
        // Touch cache lines across the 1.5 MB working set
        idx = (idx + PHYSICS_STRIDE) % stateBuffer.size();
        float s = stateBuffer[idx];

        double x = std::sin(j + s) * 1.5;
        double y = std::cos(j + s) * 1.5;
        double z = std::tan(j) * 0.5;
        double val = std::sqrt(x * x + y * y + z * z);

        stateBuffer[idx] = static_cast<float>(val * 0.001);
        sum += val;
    }
}

std::vector<double> timeFrames(std::vector<float>& buffer, long long loops, int warmupFrames, int frames) {
    for (int i = 0; i < warmupFrames; ++i) executePhysicsMath(buffer, loops);
    std::vector<double> out;
    out.reserve(static_cast<size_t>(frames));
    for (int i = 0; i < frames; ++i) {
        const auto t0 = Clock::now();
        executePhysicsMath(buffer, loops);
        out.push_back(msBetween(t0, Clock::now()));
    }
    return out;
}

double medianOf(std::vector<double> v) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// -----------------------------------------------------------------------------------------------------
// Persistent background workers (one per P-core LP except LP0), started once, joined at exit
// -----------------------------------------------------------------------------------------------------
struct Worker {
    int lp = -1;
    DWORD_PTR mask = 0;
    std::atomic<bool> pauseRequested{ false };
    std::atomic<bool> parked{ false };
    std::thread thread;
};
std::vector<std::unique_ptr<Worker>> g_workers;

void workerThreadFunc(Worker* w) {
    pinCurrentThreadOrDie(w->mask, "worker on LP " + std::to_string(w->lp));
    std::vector<float> buffer(PHYSICS_BUFFER_ELEMENTS, 1.0f);
    while (!g_shutdown.load(std::memory_order_acquire)) {
        if (w->pauseRequested.load(std::memory_order_acquire)) {
            w->parked.store(true, std::memory_order_release);
            w->pauseRequested.wait(true, std::memory_order_acquire); // sleeps (no CPU use) while parked
            w->parked.store(false, std::memory_order_release);
            continue;
        }
        executePhysicsMath(buffer, g_loopCount.load(std::memory_order_relaxed));
    }
}

void startWorkers() {
    for (int lp = 0; lp < 64; ++lp) {
        if (!(g_topo.workerMask & lpBit(lp))) continue;
        auto w = std::make_unique<Worker>();
        w->lp = lp;
        w->mask = lpBit(lp);
        g_workers.push_back(std::move(w));
    }
    for (auto& w : g_workers) w->thread = std::thread(workerThreadFunc, w.get());
}

Worker* findWorker(int lp) {
    for (auto& w : g_workers) if (w->lp == lp) return w.get();
    return nullptr;
}

void setWorkerParked(Worker& w, bool park) {
    if (park) {
        w.pauseRequested.store(true, std::memory_order_release);
        if (!waitUntilTrue([&] { return w.parked.load(std::memory_order_acquire); }, PARK_TIMEOUT_MS))
            fatal("Worker on LP " + std::to_string(w.lp) + " did not park within " + std::to_string(PARK_TIMEOUT_MS) + " ms");
    }
    else {
        w.pauseRequested.store(false, std::memory_order_release);
        w.pauseRequested.notify_all();
        if (!waitUntilTrue([&] { return !w.parked.load(std::memory_order_acquire); }, PARK_TIMEOUT_MS))
            fatal("Worker on LP " + std::to_string(w.lp) + " did not resume within " + std::to_string(PARK_TIMEOUT_MS) + " ms");
    }
}

// Parks every worker whose LP is not in activeMask and resumes every worker whose LP is.
void applyWorkerSet(DWORD_PTR activeMask) {
    for (auto& w : g_workers) {
        const bool wantActive = (activeMask & w->mask) != 0;
        const bool isParked = w->pauseRequested.load(std::memory_order_acquire);
        if (wantActive && isParked) {
            setWorkerParked(*w, false);
            std::cout << "  [Workers] resumed worker on LP " << w->lp << "\n";
        }
        else if (!wantActive && !isParked) {
            setWorkerParked(*w, true);
            std::cout << "  [Workers] parked worker on LP " << w->lp << "\n";
        }
    }
}

void stopWorkers() {
    g_shutdown.store(true, std::memory_order_release);
    for (auto& w : g_workers) {
        w->pauseRequested.store(false, std::memory_order_release);
        w->pauseRequested.notify_all();
    }
    for (auto& w : g_workers) if (w->thread.joinable()) w->thread.join();
}

// For UNPINNED the AI thread is not pinned; its mask is the process affinity it keeps (verified by the thread).
DWORD_PTR aiMaskFor(Condition c) {
    switch (c) {
    case Condition::BASELINE:          return 0;
    case Condition::P_SHARED:          return lpBit(g_topo.gameLP);
    case Condition::P_SMT:             return g_topo.siblingLP >= 0 ? lpBit(g_topo.siblingLP) : 0;
    case Condition::E_CORE:            return g_topo.eCoreAIMask;
    case Condition::SMT_IDLE:          return 0;
    case Condition::E_CORE_SMT_IDLE:   return g_topo.eCoreAIMask;
    case Condition::P_SHARED_SMT_IDLE: return lpBit(g_topo.gameLP);
    case Condition::E_CORE_1LP:        return g_topo.eCoreAIMask ? lpBit(lowestLP(g_topo.eCoreAIMask)) : 0;
    case Condition::UNPINNED:          return g_topo.processMask;
    }
    return 0;
}

DWORD_PTR workerSetFor(Condition c) {
    if (parksSibling(c) && g_topo.siblingLP >= 0) return g_topo.workerMask & ~lpBit(g_topo.siblingLP);
    return g_topo.workerMask;
}

// -----------------------------------------------------------------------------------------------------
// Topology discovery
// -----------------------------------------------------------------------------------------------------
bool setupTopology(std::string& err) {
    const WORD groups = GetActiveProcessorGroupCount();
    if (groups != 1) {
        err = "this benchmark supports a single processor group; found " + std::to_string(groups);
        return false;
    }
    DWORD_PTR procMask = 0, sysMask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &procMask, &sysMask)) {
        err = "GetProcessAffinityMask failed, GetLastError=" + std::to_string(GetLastError());
        return false;
    }
    g_topo.processMask = procMask;
    g_topo.logicalCount = static_cast<int>(GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));

    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0) {
        err = "GetLogicalProcessorInformationEx(size query) failed, GetLastError=" + std::to_string(GetLastError());
        return false;
    }
    std::vector<BYTE> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len)) {
        err = "GetLogicalProcessorInformationEx failed, GetLastError=" + std::to_string(GetLastError());
        return false;
    }

    struct CoreInfo { DWORD_PTR mask; int effClass; };
    std::vector<CoreInfo> cores;
    for (BYTE* p = buf.data(); p < buf.data() + len;) {
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(p);
        if (info->Relationship == RelationProcessorCore) {
            if (info->Processor.GroupCount != 1 || info->Processor.GroupMask[0].Group != 0) {
                err = "a processor core spans more than one group or is not in group 0";
                return false;
            }
            cores.push_back({ static_cast<DWORD_PTR>(info->Processor.GroupMask[0].Mask),
                              static_cast<int>(info->Processor.EfficiencyClass) });
        }
        p += info->Size;
    }
    if (cores.empty()) { err = "no processor cores reported"; return false; }
    std::sort(cores.begin(), cores.end(), [](const CoreInfo& a, const CoreInfo& b) { return lowestLP(a.mask) < lowestLP(b.mask); });

    int maxEff = -1, minEff = 1 << 30;
    for (const auto& c : cores) { maxEff = std::max(maxEff, c.effClass); minEff = std::min(minEff, c.effClass); }
    g_topo.hybrid = (maxEff != minEff);

    DWORD_PTR firstPCore = 0;
    for (const auto& c : cores) {
        if (c.effClass == maxEff) {
            g_topo.pCoreMask |= c.mask;
            ++g_topo.pPhysical;
            if (firstPCore == 0) firstPCore = c.mask;
        }
        else if (c.effClass == minEff) {
            g_topo.eCoreMask |= c.mask;
            ++g_topo.ePhysical;
        }
        else {
            ++g_topo.otherPhysical;
        }
    }

    g_topo.gameLP = lowestLP(firstPCore);
    const DWORD_PTR siblingBits = firstPCore & ~lpBit(g_topo.gameLP);
    g_topo.siblingLP = siblingBits ? lowestLP(siblingBits) : -1;
    g_topo.workerMask = g_topo.pCoreMask & ~lpBit(g_topo.gameLP);

    if (g_topo.hybrid) {
        if (lpCount(g_topo.eCoreMask) >= 2) {
            // Reserve the highest E-core LP for the telemetry and orchestrator threads so they never share
            // an LP with the E_CORE AI thread.
            g_topo.housekeepingLP = highestLP(g_topo.eCoreMask);
            g_topo.eCoreAIMask = g_topo.eCoreMask & ~lpBit(g_topo.housekeepingLP);
        }
        else {
            g_topo.housekeepingLP = -1;
            g_topo.eCoreAIMask = g_topo.eCoreMask;
        }
    }
    else {
        // Non-hybrid fallback: every core is a "P-core"; there is no E-core mask, no housekeeping LP and the
        // E_CORE condition is removed from the plan by main().
        g_topo.eCoreMask = 0;
        g_topo.eCoreAIMask = 0;
        g_topo.housekeepingLP = -1;
    }

    const DWORD_PTR allUsed = g_topo.pCoreMask | g_topo.eCoreMask;
    if ((allUsed & ~g_topo.processMask) != 0) {
        err = "process affinity mask " + hexMask(g_topo.processMask) + " does not include all cores " + hexMask(allUsed) +
            " (was the process started with a restricted affinity?)";
        return false;
    }
    if (g_topo.gameLP < 0) { err = "could not determine the game LP"; return false; }
    return true;
}

void printTopology() {
    std::cout << "Topology: " << g_topo.logicalCount << " logical processors, hybrid=" << (g_topo.hybrid ? "yes" : "no")
        << " (P physical " << g_topo.pPhysical << ", E physical " << g_topo.ePhysical << ")\n"
        << "  P-core LPs   : " << hexMask(g_topo.pCoreMask) << " [" << lpList(g_topo.pCoreMask) << "]\n"
        << "  E-core LPs   : " << hexMask(g_topo.eCoreMask) << " [" << lpList(g_topo.eCoreMask) << "]\n"
        << "  Game LP (LP0): " << g_topo.gameLP << " | SMT sibling: "
        << (g_topo.siblingLP >= 0 ? std::to_string(g_topo.siblingLP) : std::string("none")) << "\n"
        << "  Worker LPs   : " << hexMask(g_topo.workerMask) << " [" << lpList(g_topo.workerMask) << "]\n"
        << "  Housekeeping : " << (g_topo.housekeepingLP >= 0 ? "LP " + std::to_string(g_topo.housekeepingLP) : std::string("none"))
        << " | E_CORE AI mask: " << hexMask(g_topo.eCoreAIMask) << " [" << lpList(g_topo.eCoreAIMask) << "]\n";
    if (g_topo.otherPhysical > 0)
        std::cout << "  [WARN] " << g_topo.otherPhysical << " cores with an intermediate EfficiencyClass are not used.\n";
    std::cout << "\n";
}

// -----------------------------------------------------------------------------------------------------
// Statistics
// -----------------------------------------------------------------------------------------------------
struct SampleStats {
    size_t n = 0;
    double mean = std::numeric_limits<double>::quiet_NaN();
    double median = std::numeric_limits<double>::quiet_NaN();
    double min = std::numeric_limits<double>::quiet_NaN();
    double p99 = std::numeric_limits<double>::quiet_NaN();
    double p999 = std::numeric_limits<double>::quiet_NaN();
    double max = std::numeric_limits<double>::quiet_NaN();
    double stddev = std::numeric_limits<double>::quiet_NaN();
};

// Nearest-rank percentile of sorted data: the smallest sample such that at least num/den of all samples are
// <= it. rank = ceil(n * num / den) (1-based), computed in exact integer arithmetic.
double nearestRank(const std::vector<double>& sorted, uint64_t num, uint64_t den) {
    if (sorted.empty()) return std::numeric_limits<double>::quiet_NaN();
    const uint64_t n = sorted.size();
    const uint64_t rank = std::clamp<uint64_t>((n * num + den - 1) / den, 1, n);
    return sorted[rank - 1];
}

SampleStats computeStats(std::vector<double> v) {
    SampleStats s;
    s.n = v.size();
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    const double n = static_cast<double>(v.size());
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
    s.min = v.front();
    s.max = v.back();
    s.median = (v.size() % 2) ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    s.p99 = nearestRank(v, 99, 100);
    s.p999 = nearestRank(v, 999, 1000);
    if (v.size() >= 2) {
        double ss = 0.0;
        for (double x : v) ss += (x - s.mean) * (x - s.mean);
        s.stddev = std::sqrt(ss / (n - 1.0)); // sample standard deviation
    }
    return s;
}

// -----------------------------------------------------------------------------------------------------
// Telemetry (PDH)
// -----------------------------------------------------------------------------------------------------
struct Metric {
    double peak = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    long long n = 0;
    void add(double v) {
        if (!std::isfinite(v)) return;
        peak = std::max(peak, v);
        sum += v;
        ++n;
    }
    double peakOr(double fallback) const { return n ? peak : fallback; }
    double meanOr(double fallback) const { return n ? sum / static_cast<double>(n) : fallback; }
};

struct TelemetryAccum {
    long long samples = 0;
    Metric thermalC, powerMeterW, raplPkgW, perfP, perfE, pageFaults, fileCacheFaults;
};

enum class ProcPerfMethod { Unavailable, PerLP, Total };

bool readCounterArray(PDH_HCOUNTER c, std::vector<std::pair<std::wstring, double>>& out) {
    out.clear();
    if (!c) return false;
    // PDH_FMT_NOCAP100: without it PDH clamps percentage counters (e.g. % Processor Performance) to 100.
    const DWORD fmt = PDH_FMT_DOUBLE | PDH_FMT_NOCAP100;
    for (int attempt = 0; attempt < 3; ++attempt) {
        DWORD bytes = 0, count = 0;
        PDH_STATUS st = PdhGetFormattedCounterArrayW(c, fmt, &bytes, &count, nullptr);
        if (st != PDH_MORE_DATA || bytes == 0) return false;
        std::vector<BYTE> buf(bytes);
        auto* items = reinterpret_cast<PPDH_FMT_COUNTERVALUE_ITEM_W>(buf.data());
        st = PdhGetFormattedCounterArrayW(c, fmt, &bytes, &count, items);
        if (st == PDH_MORE_DATA) continue; // instance set changed between calls
        if (st != ERROR_SUCCESS) return false;
        for (DWORD i = 0; i < count; ++i) {
            const DWORD cs = items[i].FmtValue.CStatus;
            if (cs == PDH_CSTATUS_VALID_DATA || cs == PDH_CSTATUS_NEW_DATA)
                out.emplace_back(items[i].szName ? items[i].szName : L"", items[i].FmtValue.doubleValue);
        }
        return !out.empty();
    }
    return false;
}

bool readCounterSingle(PDH_HCOUNTER c, double& v) {
    if (!c) return false;
    PDH_FMT_COUNTERVALUE fv{};
    if (PdhGetFormattedCounterValue(c, PDH_FMT_DOUBLE | PDH_FMT_NOCAP100, nullptr, &fv) != ERROR_SUCCESS) return false;
    if (fv.CStatus != PDH_CSTATUS_VALID_DATA && fv.CStatus != PDH_CSTATUS_NEW_DATA) return false;
    v = fv.doubleValue;
    return true;
}

class PdhSampler {
public:
    PdhSampler() = default;
    PdhSampler(const PdhSampler&) = delete;
    PdhSampler& operator=(const PdhSampler&) = delete;
    ~PdhSampler() { close(); }

    // Opens the query and adds every counter, checking each PdhAddEnglishCounterW return code. Counters that
    // cannot be added are reported in 'problems' and later written as -1.
    bool open(std::vector<std::string>* problems) {
        const PDH_STATUS st = PdhOpenQueryW(nullptr, 0, &query_);
        if (st != ERROR_SUCCESS) {
            query_ = nullptr;
            if (problems) problems->push_back("PdhOpenQueryW failed: " + pdhStatusHex(st));
            return false;
        }
        auto add = [&](const wchar_t* path, PDH_HCOUNTER& h) {
            const PDH_STATUS s = PdhAddEnglishCounterW(query_, path, 0, &h);
            if (s != ERROR_SUCCESS) {
                h = nullptr;
                if (problems) problems->push_back("PdhAddEnglishCounterW(" + narrow(path) + ") failed: " + pdhStatusHex(s));
                return false;
            }
            return true;
        };
        // ACPI thermal zone: "High Precision Temperature" is in tenths of Kelvin, "Temperature" in whole Kelvin.
        if (add(L"\\Thermal Zone Information(*)\\High Precision Temperature", thermal_)) {
            thermalLabel_ = "High Precision Temperature (0.1 K)";
            thermalTenthsK_ = true;
        }
        else if (add(L"\\Thermal Zone Information(*)\\Temperature", thermal_)) {
            thermalLabel_ = "Temperature (K)";
            thermalTenthsK_ = false;
        }
        add(L"\\Power Meter(*)\\Power", powerMeter_);                     // milliwatts
        add(L"\\Energy Meter(RAPL_Package0_PKG)\\Power", rapl_);           // milliwatts
        add(L"\\Processor Information(*)\\% Processor Performance", procPerf_);
        add(L"\\Memory\\Page Faults/sec", pageFaults_);
        add(L"\\Memory\\Cache Faults/sec", cacheFaults_);                   // file-system cache faults
        return true;
    }

    bool collect() { return query_ != nullptr && PdhCollectQueryData(query_) == ERROR_SUCCESS; }

    void read(TelemetryAccum& acc, ProcPerfMethod* method = nullptr) const {
        std::vector<std::pair<std::wstring, double>> items;
        double v = 0.0;

        if (readCounterArray(thermal_, items)) {
            double hottest = -std::numeric_limits<double>::infinity();
            for (const auto& [name, raw] : items) {
                const double c = thermalTenthsK_ ? raw / 10.0 - 273.15 : raw - 273.15;
                if (c > -50.0 && c < 150.0) hottest = std::max(hottest, c);
            }
            acc.thermalC.add(hottest);
        }

        if (readCounterArray(powerMeter_, items)) {
            double total = std::numeric_limits<double>::quiet_NaN(), sum = 0.0;
            bool any = false;
            for (const auto& [name, mw] : items) {
                if (name == L"_Total") total = mw;
                else { sum += mw; any = true; }
            }
            const double mw = std::isfinite(total) ? total : (any ? sum : std::numeric_limits<double>::quiet_NaN());
            acc.powerMeterW.add(mw / 1000.0);
        }

        if (readCounterSingle(rapl_, v)) acc.raplPkgW.add(v / 1000.0);

        // Instance names are "<group>,<lp>", plus "<group>,_Total" and "_Total".
        if (readCounterArray(procPerf_, items)) {
            double sumP = 0.0, sumE = 0.0, total = std::numeric_limits<double>::quiet_NaN();
            int nP = 0, nE = 0;
            for (const auto& [name, val] : items) {
                if (name == L"_Total") { total = val; continue; }
                int group = -1, lp = -1;
                if (swscanf_s(name.c_str(), L"%d,%d", &group, &lp) == 2 && group == 0 && lp >= 0 && lp < 64) {
                    if (g_topo.pCoreMask & lpBit(lp)) { sumP += val; ++nP; }
                    else if (g_topo.eCoreMask & lpBit(lp)) { sumE += val; ++nE; }
                }
            }
            if (nP > 0) {
                acc.perfP.add(sumP / nP);
                if (nE > 0) acc.perfE.add(sumE / nE);
                if (method) *method = ProcPerfMethod::PerLP;
            }
            else if (std::isfinite(total)) {
                acc.perfP.add(total);
                acc.perfE.add(total);
                if (method) *method = ProcPerfMethod::Total;
            }
        }

        if (readCounterSingle(pageFaults_, v)) acc.pageFaults.add(v);
        if (readCounterSingle(cacheFaults_, v)) acc.fileCacheFaults.add(v);
    }

    void close() {
        if (query_) { PdhCloseQuery(query_); query_ = nullptr; }
    }

    const std::string& thermalLabel() const { return thermalLabel_; }

private:
    PDH_HQUERY query_ = nullptr;
    PDH_HCOUNTER thermal_ = nullptr;
    PDH_HCOUNTER powerMeter_ = nullptr;
    PDH_HCOUNTER rapl_ = nullptr;
    PDH_HCOUNTER procPerf_ = nullptr;
    PDH_HCOUNTER pageFaults_ = nullptr;
    PDH_HCOUNTER cacheFaults_ = nullptr;
    bool thermalTenthsK_ = false;
    std::string thermalLabel_ = "unavailable";
};

struct TelemetryProbe {
    std::string thermal = "unavailable";
    bool powerMeter = false, rapl = false, procPerf = false, pageFaults = false, fileCacheFaults = false;
    ProcPerfMethod procPerfMethod = ProcPerfMethod::Unavailable;
    std::vector<std::string> problems;
};

// One-off probe at startup: reports which counters exist and how % Processor Performance is split.
TelemetryProbe probeTelemetry() {
    TelemetryProbe r;
    PdhSampler s;
    if (!s.open(&r.problems)) return r;
    s.collect();
    Sleep(250);
    s.collect();
    TelemetryAccum acc;
    s.read(acc, &r.procPerfMethod);
    r.thermal = acc.thermalC.n ? s.thermalLabel() : "unavailable";
    r.powerMeter = acc.powerMeterW.n > 0;
    r.rapl = acc.raplPkgW.n > 0;
    r.procPerf = acc.perfP.n > 0;
    r.pageFaults = acc.pageFaults.n > 0;
    r.fileCacheFaults = acc.fileCacheFaults.n > 0;
    return r;
}

const char* procPerfMethodName(ProcPerfMethod m) {
    switch (m) {
    case ProcPerfMethod::PerLP: return "per-LP instances split by topology (unweighted mean per cluster)";
    case ProcPerfMethod::Total: return "_Total only (P and E columns both hold _Total)";
    default: return "unavailable";
    }
}

struct TelemetryControl {
    std::atomic<bool> ready{ false };
    std::atomic<bool> begin{ false };
    std::atomic<bool> stop{ false };
};

void telemetryThreadFunc(TelemetryControl* ctl, TelemetryAccum* acc, int intervalMs) {
    if (g_topo.housekeepingLP >= 0) pinCurrentThreadOrDie(lpBit(g_topo.housekeepingLP), "telemetry thread");
    PdhSampler sampler;
    const bool ok = sampler.open(nullptr); // failures were already reported by the startup probe
    if (ok) sampler.collect();             // opened and primed during warm-up, outside the window
    ctl->ready.store(true, std::memory_order_release);

    while (!ctl->begin.load(std::memory_order_acquire) && !ctl->stop.load(std::memory_order_acquire)) Sleep(1);
    if (!ok || ctl->stop.load(std::memory_order_acquire)) return;

    sampler.collect(); // fresh baseline at window start: every rate sample covers only the measurement window
    for (;;) {
        Sleep(static_cast<DWORD>(intervalMs));
        if (ctl->stop.load(std::memory_order_acquire)) break; // never sample past the end of the window
        if (!sampler.collect()) continue;
        sampler.read(*acc);
        ++acc->samples;
    }
}

// -----------------------------------------------------------------------------------------------------
// Per-test synchronization
// -----------------------------------------------------------------------------------------------------
struct TestSync {
    std::atomic<bool> gameWarm{ false };
    std::atomic<bool> aiWarm{ false };
    std::atomic<bool> aiFailed{ false };
    std::atomic<bool> measuring{ false };
    std::atomic<bool> stop{ false };
    std::atomic<long long> stopTicks{ 0 };
    std::atomic<long long> gameFrames{ 0 };
    std::atomic<long long> aiBatches{ 0 };
};

void requestStop(TestSync& s) {
    bool expected = false;
    const long long now = Clock::now().time_since_epoch().count();
    if (s.stop.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        s.stopTicks.store(now, std::memory_order_release);
    }
}

// Wakes any thread still waiting for the measurement window (used after stop has been requested).
void releaseWaiters(TestSync& s) {
    s.measuring.store(true, std::memory_order_release);
    s.measuring.notify_all();
}

// -----------------------------------------------------------------------------------------------------
// Game thread (LP0)
// -----------------------------------------------------------------------------------------------------
// Frame start and end times as steady_clock ticks, for the --trace frame rows.
struct FrameTrace {
    std::vector<long long> start, end;
};

void gameThreadFunc(TestSync* sync, std::vector<double>* frames, FrameTrace* trace, size_t capacity) {
    pinCurrentThreadOrDie(lpBit(g_topo.gameLP), "game thread");
    std::vector<float> physics(PHYSICS_BUFFER_ELEMENTS, 1.0f);
    frames->assign(capacity, 0.0); // pre-touched so no page faults occur inside the measurement window
    trace->start.assign(capacity, 0);
    trace->end.assign(capacity, 0);
    const long long loops = g_loopCount.load(std::memory_order_relaxed);

    for (int i = 0; i < GAME_WARMUP_FRAMES; ++i) executePhysicsMath(physics, loops);
    sync->gameWarm.store(true, std::memory_order_release);

    // Keep LP0 busy with unrecorded frames while the AI thread loads its model and warms up.
    while (!sync->measuring.load(std::memory_order_acquire) && !sync->stop.load(std::memory_order_acquire))
        executePhysicsMath(physics, loops);

    size_t count = 0;
    while (!sync->stop.load(std::memory_order_acquire)) {
        const auto t0 = Clock::now();
        executePhysicsMath(physics, loops);
        const auto t1 = Clock::now();
        if (sync->stop.load(std::memory_order_acquire)) break; // frame overlapped the end of the window
        trace->start[count] = t0.time_since_epoch().count();
        trace->end[count] = t1.time_since_epoch().count();
        (*frames)[count++] = msBetween(t0, t1);
        sync->gameFrames.store(static_cast<long long>(count), std::memory_order_release);
        if (count == capacity) { requestStop(*sync); break; }
    }
    frames->resize(count);
    trace->start.resize(count);
    trace->end.resize(count);
}

// -----------------------------------------------------------------------------------------------------
// AI thread
// -----------------------------------------------------------------------------------------------------
struct AIParams {
    Ort::Env* env = nullptr;
    std::wstring modelPath;
    int inputs = 0;
    int agents = 0;
    DWORD_PTR mask = 0;
    bool pin = true;                 // false (UNPINNED): keep the process affinity, which must equal 'mask'
    double tickMs = 50.0;
    int maxBatches = 0;
};

// One measured batch, for the --trace batch rows. Times are steady_clock ticks. The LP and cycle reads are
// taken outside [start, end], so the batch latency is measured exactly as without them.
struct BatchTrace {
    long long release = 0;           // scheduled start of the batch's update period
    long long start = 0;             // when the AI thread started the batch (after waking)
    long long end = 0;               // when the batch finished
    int lpStart = -1, lpEnd = -1;    // LP the thread was on just before and just after the batch
    unsigned long long cycles = 0;   // QueryThreadCycleTime difference over the batch
    bool missed = false;             // ended after release + period
};

struct AIResult {
    std::vector<double> batchMs;
    std::vector<BatchTrace> trace;
    long long missedDeadlines = 0;
    double busyMs = 0.0;
    double outputSum = 0.0;
    std::string batchMode = "unknown";
    std::string warning;
    std::string error;
};

int currentLP() {
    PROCESSOR_NUMBER pn{};
    GetCurrentProcessorNumberEx(&pn);
    return pn.Group == 0 ? static_cast<int>(pn.Number) : -1;
}

unsigned long long threadCycles() {
    ULONG64 c = 0;
    QueryThreadCycleTime(GetCurrentThread(), &c);
    return c;
}

// Sleeps until 'deadline' with a high-resolution waitable timer (relative due times computed from the
// absolute QPC deadline). Without such a timer it falls back to Sleep() under timeBeginPeriod(1), which may
// overshoot by ~1 ms but never spins.
void sleepUntil(Clock::time_point deadline, HANDLE timer) {
    for (;;) {
        const auto now = Clock::now();
        if (now >= deadline) return;
        const long long remain100ns = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now).count() / 100;
        if (timer) {
            LARGE_INTEGER due{};
            due.QuadPart = -std::max<long long>(remain100ns, 1); // negative = relative, in 100 ns units
            if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE) &&
                WaitForSingleObject(timer, INFINITE) == WAIT_OBJECT_0) {
                continue;
            }
            timer = nullptr; // unexpected failure: use the Sleep fallback for the rest of this wait
        }
        const DWORD ms = static_cast<DWORD>((remain100ns + 9999) / 10000);
        Sleep(std::max<DWORD>(ms, 1));
        return;
    }
}

// ONNX Runtime session options of every timed AI thread. --verify-outputs uses the same function, so the
// verified outputs come from exactly the configuration that was timed.
Ort::SessionOptions makeSessionOptions() {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(1);
    so.SetInterOpNumThreads(1);
    so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    so.AddConfigEntry("session.set_denormal_as_zero", "1");
    so.AddConfigEntry("session.intra_op.allow_spinning", "0");
    so.AddConfigEntry("session.inter_op.allow_spinning", "0");
    return so;
}

// Fixed synthetic agent inputs of every timed batch: 0.5, with every 16th value offset by (i % 100) * 0.001.
void fillBenchmarkInputs(std::vector<float>& buffer) {
    std::fill(buffer.begin(), buffer.end(), 0.5f);
    for (size_t i = 0; i < buffer.size(); i += 16) buffer[i] = 0.5f + static_cast<float>(i % 100) * 0.001f;
}

void aiThreadFunc(AIParams p, TestSync* sync, AIResult* out) {
    if (p.pin) {
        pinCurrentThreadOrDie(p.mask, "AI thread");
    }
    else {
        // UNPINNED: no affinity call. Verify that the thread kept the process affinity, so Windows can choose any LP.
        GROUP_AFFINITY ga{};
        if (!GetThreadGroupAffinity(GetCurrentThread(), &ga))
            fatal("GetThreadGroupAffinity failed for the unpinned AI thread, GetLastError=" + std::to_string(GetLastError()));
        if (ga.Group != 0 || static_cast<DWORD_PTR>(ga.Mask) != p.mask)
            fatal("Unpinned AI thread: inherited affinity is group " + std::to_string(ga.Group) + " mask " +
                hexMask(static_cast<DWORD_PTR>(ga.Mask)) + ", expected the process mask " + hexMask(p.mask));
    }

    // Flush-to-zero / denormals-are-zero for this thread. ONNX Runtime applies "session.set_denormal_as_zero"
    // (also set below) only to the thread that creates the FIRST session in the process and to ORT pool
    // threads. This benchmark creates a new AI thread per test and runs without an intra-op pool, so MXCSR is
    // set explicitly to keep the floating-point mode identical in every test.
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    HANDLE timer = nullptr;
    bool localTimePeriod = false;
    if (g_highResTimerAvailable) {
        timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    }
    if (!timer && g_highResTimerAvailable) {
        timeBeginPeriod(1); // the process-wide 1 ms period is only active when no high-res timer exists at all
        localTimePeriod = true;
    }

    try {
        Ort::SessionOptions so = makeSessionOptions();
        Ort::Session session(*p.env, p.modelPath.c_str(), so);

        if (session.GetInputCount() != 1 || session.GetOutputCount() < 1)
            throw std::runtime_error("unexpected model I/O count");
        Ort::AllocatorWithDefaultOptions allocator;
        const std::string inName = session.GetInputNameAllocated(0, allocator).get();
        const std::string outName = session.GetOutputNameAllocated(0, allocator).get();
        const char* inNames[] = { inName.c_str() };
        const char* outNames[] = { outName.c_str() };

        const std::vector<int64_t> inShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (inShape.size() != 3)
            throw std::runtime_error("model input rank is " + std::to_string(inShape.size()) + ", expected 3 [batch, seq, features]");
        if ((inShape[1] >= 0 && inShape[1] != SEQ_LEN) || (inShape[2] >= 0 && inShape[2] != p.inputs))
            throw std::runtime_error("model input shape does not match [batch, " + std::to_string(SEQ_LEN) + ", " + std::to_string(p.inputs) + "]");

        // Full contiguous working set for all agents, so the cache footprint scales with the agent count.
        const size_t perAgent = static_cast<size_t>(SEQ_LEN) * static_cast<size_t>(p.inputs);
        std::vector<float> agentBuffer(static_cast<size_t>(p.agents) * perAgent);
        fillBenchmarkInputs(agentBuffer);

        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> batchedShape = { p.agents, SEQ_LEN, p.inputs };
        std::vector<int64_t> singleShape = { 1, SEQ_LEN, p.inputs };

        // Batched mode if the model accepts [agents, seq, features]; otherwise one Run() per agent.
        Ort::Value batchedTensor{ nullptr };
        bool batched = false;
        std::string fallbackReason;
        if (inShape[0] < 0 || inShape[0] == p.agents) {
            try {
                batchedTensor = Ort::Value::CreateTensor<float>(memInfo, agentBuffer.data(), agentBuffer.size(),
                    batchedShape.data(), batchedShape.size());
                auto probe = session.Run(Ort::RunOptions{ nullptr }, inNames, &batchedTensor, 1, outNames, 1);
                const size_t n = probe.front().GetTensorTypeAndShapeInfo().GetElementCount();
                if (n == static_cast<size_t>(p.agents) * 3) batched = true;
                else fallbackReason = "batched output has " + std::to_string(n) + " elements, expected " + std::to_string(p.agents * 3);
            }
            catch (const Ort::Exception& e) {
                fallbackReason = std::string("batched probe failed: ") + e.what();
            }
        }
        else {
            fallbackReason = "input batch dimension is fixed at " + std::to_string(inShape[0]);
        }
        out->batchMode = batched ? "batched" : "per_agent";
        if (!batched) out->warning = "model fell back to per-agent mode (" + fallbackReason + ")";

        auto sumOutputs = [](const Ort::Value& v) {
            const size_t n = v.GetTensorTypeAndShapeInfo().GetElementCount();
            const float* d = v.GetTensorData<float>();
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) s += d[i];
            return s;
        };
        auto runBatch = [&]() -> double {
            if (batched) {
                auto outs = session.Run(Ort::RunOptions{ nullptr }, inNames, &batchedTensor, 1, outNames, 1);
                return sumOutputs(outs.front());
            }
            double s = 0.0;
            for (int a = 0; a < p.agents; ++a) {
                float* slice = agentBuffer.data() + static_cast<size_t>(a) * perAgent;
                Ort::Value t = Ort::Value::CreateTensor<float>(memInfo, slice, perAgent, singleShape.data(), singleShape.size());
                auto outs = session.Run(Ort::RunOptions{ nullptr }, inNames, &t, 1, outNames, 1);
                s += sumOutputs(outs.front());
            }
            return s;
        };

        for (int w = 0; w < AI_WARMUP_BATCHES; ++w) runBatch();

        out->batchMs.assign(static_cast<size_t>(p.maxBatches), 0.0); // pre-touched
        out->trace.assign(static_cast<size_t>(p.maxBatches), BatchTrace{});
        size_t count = 0;

        sync->aiWarm.store(true, std::memory_order_release);
        sync->measuring.wait(false, std::memory_order_acquire); // main thread opens the window

        const auto tick = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(p.tickMs));
        Clock::time_point tickStart = Clock::now();
        while (!sync->stop.load(std::memory_order_acquire)) {
            const unsigned long long c0 = threadCycles();
            const int lp0 = currentLP();
            const auto t0 = Clock::now();
            const double outputSum = runBatch();
            const auto t1 = Clock::now();
            const int lp1 = currentLP();
            const unsigned long long c1 = threadCycles();
            if (sync->stop.load(std::memory_order_acquire)) break; // batch overlapped the end of the window

            // Absolute-deadline pacing: tick k is scheduled at tickStart_k; it must finish by tickStart_k + period.
            const auto deadline = tickStart + tick;
            BatchTrace& bt = out->trace[count];
            bt.release = tickStart.time_since_epoch().count();
            bt.start = t0.time_since_epoch().count();
            bt.end = t1.time_since_epoch().count();
            bt.lpStart = lp0;
            bt.lpEnd = lp1;
            bt.cycles = c1 - c0;
            bt.missed = t1 > deadline;

            const double ms = msBetween(t0, t1);
            out->batchMs[count++] = ms;
            out->busyMs += ms;
            out->outputSum += outputSum;
            sync->aiBatches.store(static_cast<long long>(count), std::memory_order_release);
            if (count >= static_cast<size_t>(p.maxBatches)) { requestStop(*sync); break; }

            if (t1 > deadline) {
                ++out->missedDeadlines; // overran its tick budget: start the next tick immediately
                tickStart = t1;
            }
            else {
                sleepUntil(deadline, timer);
                tickStart = deadline;   // next tick starts on the schedule, not "now + period" (no drift)
            }
        }
        out->batchMs.resize(count);
        out->trace.resize(count);
    }
    catch (const Ort::Exception& e) {
        out->error = std::string("ONNX Runtime error: ") + e.what();
    }
    catch (const std::exception& e) {
        out->error = std::string("AI thread error: ") + e.what();
    }

    if (!out->error.empty()) {
        sync->aiFailed.store(true, std::memory_order_release);
        requestStop(*sync);
    }
    if (timer) CloseHandle(timer);
    if (localTimePeriod) timeEndPeriod(1);
}

// -----------------------------------------------------------------------------------------------------
// Calibration persistence
// -----------------------------------------------------------------------------------------------------
struct CalibrationInfo {
    long long loopCount = 0;
    std::string calibratedAt;
    double measuredMedianMs = std::numeric_limits<double>::quiet_NaN();
    double siblingParkedRefMs = std::numeric_limits<double>::quiet_NaN();
    int soakSeconds = -1;
    int gameLP = -1;
};

enum class CalLoad { Ok, Missing, Legacy, Invalid };

std::string trimCopy(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

CalLoad loadCalibration(const std::string& path, CalibrationInfo& out) {
    std::ifstream f(path);
    if (!f.is_open()) return CalLoad::Missing;
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line)) {
        line = trimCopy(line);
        if (!line.empty()) lines.push_back(line);
    }
    if (lines.empty()) return CalLoad::Invalid;
    if (lines[0].find('=') == std::string::npos) return CalLoad::Legacy; // old format: a bare loop count

    CalibrationInfo c;
    try {
        for (const auto& l : lines) {
            const auto eq = l.find('=');
            if (eq == std::string::npos) continue;
            const std::string key = l.substr(0, eq), val = l.substr(eq + 1);
            if (key == "loop_count") c.loopCount = std::stoll(val);
            else if (key == "calibrated_at") c.calibratedAt = val;
            else if (key == "measured_median_ms" && !val.empty()) c.measuredMedianMs = std::stod(val);
            else if (key == "sibling_parked_ref_ms" && !val.empty()) c.siblingParkedRefMs = std::stod(val);
            else if (key == "soak_seconds") c.soakSeconds = std::stoi(val);
            else if (key == "game_lp") c.gameLP = std::stoi(val);
        }
    }
    catch (const std::exception&) {
        return CalLoad::Invalid;
    }
    if (c.loopCount < 1000 || c.calibratedAt.empty()) return CalLoad::Invalid;
    out = c;
    return CalLoad::Ok;
}

bool saveCalibration(const std::string& path, const CalibrationInfo& c) {
    std::ofstream f(path, std::ios::trunc);
    if (!f.is_open()) return false;
    auto num = [](double v) {
        if (!std::isfinite(v)) return std::string();
        std::ostringstream os;
        os << std::fixed << std::setprecision(4) << v;
        return os.str();
    };
    f << "loop_count=" << c.loopCount << "\n"
      << "calibrated_at=" << c.calibratedAt << "\n"
      << "target_ms=" << num(TARGET_FRAME_MS) << "\n"
      << "measured_median_ms=" << num(c.measuredMedianMs) << "\n"
      << "sibling_parked_ref_ms=" << num(c.siblingParkedRefMs) << "\n"
      << "soak_seconds=" << c.soakSeconds << "\n"
      << "game_lp=" << c.gameLP << "\n";
    return static_cast<bool>(f);
}

// -----------------------------------------------------------------------------------------------------
// Soak, calibration and drift check (run on the main thread, temporarily pinned to LP0)
// -----------------------------------------------------------------------------------------------------
struct SoakSummary {
    int seconds = 0;
    double firstWindowMedianMs = std::numeric_limits<double>::quiet_NaN();
    double lastWindowMedianMs = std::numeric_limits<double>::quiet_NaN();
    bool aborted = false;
};

SoakSummary runSoak(int seconds, std::vector<float>& buffer) {
    SoakSummary s;
    s.seconds = seconds;
    if (seconds <= 0) return s;
    const DWORD_PTR prev = pinCurrentThreadOrDie(lpBit(g_topo.gameLP), "main thread (soak)");
    const long long loops = g_loopCount.load();
    std::cout << "[Soak] " << seconds << " s under full worker load (game frames on LP " << g_topo.gameLP
        << ", loop count " << loops << ")...\n";

    const auto start = Clock::now();
    auto windowStart = start;
    std::vector<double> window;
    for (;;) {
        if (g_abortRequested.load()) { s.aborted = true; break; }
        const auto t0 = Clock::now();
        executePhysicsMath(buffer, loops);
        const auto t1 = Clock::now();
        window.push_back(msBetween(t0, t1));
        const double elapsed = std::chrono::duration<double>(t1 - start).count();
        if (std::chrono::duration<double>(t1 - windowStart).count() >= SOAK_REPORT_WINDOW_S || elapsed >= seconds) {
            const double med = medianOf(window);
            if (!std::isfinite(s.firstWindowMedianMs)) s.firstWindowMedianMs = med;
            s.lastWindowMedianMs = med;
            std::cout << "  [Soak] t=" << std::setw(4) << static_cast<int>(elapsed) << " s | median frame "
                << med << " ms | " << std::max(0, seconds - static_cast<int>(elapsed)) << " s left\n";
            window.clear();
            windowStart = t1;
        }
        if (elapsed >= seconds) break;
    }
    pinCurrentThreadOrDie(prev, "main thread (restore after soak)");
    return s;
}

bool runCalibration(std::vector<float>& buffer, int soakSeconds, CalibrationInfo& c) {
    const DWORD_PTR prev = pinCurrentThreadOrDie(lpBit(g_topo.gameLP), "main thread (calibration)");
    std::cout << "[Calibration] Calibrating the physics frame to " << TARGET_FRAME_MS << " ms (median of "
        << CALIBRATION_FRAMES << " frames) on LP " << g_topo.gameLP << " under load...\n";

    long long loops = g_loopCount.load();
    bool converged = false;
    for (int it = 0; it < CALIBRATION_MAX_ITERS && !g_abortRequested.load(); ++it) {
        const double med = medianOf(timeFrames(buffer, loops, 2, CALIBRATION_FRAMES));
        std::cout << "  iter " << std::setw(2) << it + 1 << ": loops=" << loops << " -> median " << med << " ms\n";
        if (std::abs(med - TARGET_FRAME_MS) <= CALIBRATION_TOLERANCE_MS) { converged = true; break; }
        loops = std::max<long long>(1000, std::llround(static_cast<double>(loops) * TARGET_FRAME_MS / std::max(0.01, med)));
    }
    if (g_abortRequested.load()) {
        pinCurrentThreadOrDie(prev, "main thread (restore after calibration)");
        return false;
    }
    if (!converged) std::cout << "  [WARN] calibration did not converge within " << CALIBRATION_MAX_ITERS << " iterations; using the last estimate\n";

    c.loopCount = loops;
    c.measuredMedianMs = medianOf(timeFrames(buffer, loops, 3, VERIFY_FRAMES));

    // Reference frame time with LP_sib's worker parked: the drift reference for P_SMT tests, whose worker set
    // differs from the calibration worker set.
    if (Worker* w = (g_topo.siblingLP >= 0) ? findWorker(g_topo.siblingLP) : nullptr) {
        setWorkerParked(*w, true);
        c.siblingParkedRefMs = medianOf(timeFrames(buffer, loops, DRIFT_WARMUP_FRAMES, VERIFY_FRAMES));
        setWorkerParked(*w, false);
    }

    // Cleanup: restore the main thread's affinity after calibration.
    pinCurrentThreadOrDie(prev, "main thread (restore after calibration)");

    c.calibratedAt = isoLocalNow();
    c.soakSeconds = soakSeconds;
    c.gameLP = g_topo.gameLP;
    return true;
}

double preTestDriftCheck(std::vector<float>& buffer) {
    const DWORD_PTR prev = pinCurrentThreadOrDie(lpBit(g_topo.gameLP), "main thread (drift check)");
    const double med = medianOf(timeFrames(buffer, g_loopCount.load(), DRIFT_WARMUP_FRAMES, DRIFT_FRAMES));
    pinCurrentThreadOrDie(prev, "main thread (restore after drift check)");
    return med;
}

// -----------------------------------------------------------------------------------------------------
// CSV
// -----------------------------------------------------------------------------------------------------
const std::vector<std::string>& csvColumns() {
    static const std::vector<std::string> cols = {
        "Run_ID", "Timestamp_Start", "Timestamp_End",
        "Inputs", "Hidden_Size", "Layers", "Agent_Density", "Condition", "Trial", "Block_Pos",
        "AI_LP_Mask", "Batch_Mode", "Calibrated_Loop_Count",
        "Pre_Test_Frame_ms", "Drift_Ref_ms", "Drift_Flag", "Truncated", "Measure_Window_s",
        "Game_Frames", "Game_Avg_ms", "Game_Median_ms", "Game_Min_ms", "Game_99th_ms", "Game_P999_ms",
        "Game_Max_ms", "Game_StdDev", "Game_P1Low_FPS",
        "AI_Batches", "AI_Avg_ms", "AI_Median_ms", "AI_99th_ms", "AI_Max_ms", "AI_Agents_Per_Sec",
        "AI_Missed_Deadlines", "AI_Duty_Cycle", "AI_Output_Checksum",
        "Telemetry_Samples",
        "Peak_ACPI_ThermalZone_C", "Mean_ACPI_ThermalZone_C",
        "Peak_PowerMeter_W", "Mean_PowerMeter_W",
        "Peak_RAPL_Package_W", "Mean_RAPL_Package_W",
        "Peak_ProcPerf_P_Pct", "Mean_ProcPerf_P_Pct", "Peak_ProcPerf_E_Pct", "Mean_ProcPerf_E_Pct",
        "Peak_PageFaults_per_s", "Peak_FileCacheFaults_per_s",
        "Trace_ID", "AI_Start_Delay_Median_ms", "AI_Start_Delay_99th_ms", "AI_Start_Delay_Max_ms",
        "AI_OnCPU_Share", "AI_Start_LPs", "AI_Batches_LP_Changed"
    };
    return cols;
}

std::string csvHeaderLine() {
    std::string h;
    for (const auto& c : csvColumns()) { if (!h.empty()) h += ","; h += c; }
    return h;
}

int csvColumnIndex(const std::string& name) {
    const auto& cols = csvColumns();
    const auto it = std::find(cols.begin(), cols.end(), name);
    return it == cols.end() ? -1 : static_cast<int>(it - cols.begin());
}

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> out;
    size_t start = 0;
    for (;;) {
        const size_t comma = line.find(',', start);
        if (comma == std::string::npos) { out.push_back(line.substr(start)); break; }
        out.push_back(line.substr(start, comma - start));
        start = comma + 1;
    }
    return out;
}

bool parseIntField(const std::string& s, int& v) {
    if (s.empty()) return false;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
    return r.ec == std::errc() && r.ptr == s.data() + s.size();
}

std::string resumeKey(int inputs, int hidden, int layers, int density, Condition c, int trial) {
    return std::to_string(inputs) + "|" + std::to_string(hidden) + "|" + std::to_string(layers) + "|" +
        std::to_string(density) + "|" + conditionName(c) + "|" + std::to_string(trial);
}

struct ResumeState {
    std::set<std::string> completed;
    size_t validRows = 0;
    size_t ignoredLines = 0;
    bool hasContent = false;
    bool endsWithNewline = true;
};

// Only newline-terminated rows with exactly the expected column count and parseable key fields count as
// completed; partial or malformed lines are ignored (and re-run).
bool loadResumeState(const std::string& path, ResumeState& st, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return true;
    const std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (content.empty()) return true;
    st.hasContent = true;
    st.endsWithNewline = content.back() == '\n';

    std::vector<std::string> lines;
    size_t pos = 0;
    for (;;) {
        const size_t nl = content.find('\n', pos);
        if (nl == std::string::npos) {
            if (pos < content.size()) ++st.ignoredLines; // trailing partial line (no newline)
            break;
        }
        std::string l = content.substr(pos, nl - pos);
        if (!l.empty() && l.back() == '\r') l.pop_back();
        lines.push_back(std::move(l));
        pos = nl + 1;
    }
    if (lines.empty()) {
        err = "'" + path + "' has no complete header line";
        return false;
    }
    if (lines[0] != csvHeaderLine()) {
        err = "'" + path + "' was written with a different column layout. Move/rename it or pass --csv <new file>.\n"
              "  expected: " + csvHeaderLine() + "\n  found:    " + lines[0];
        return false;
    }

    const size_t expected = csvColumns().size();
    const int iIn = csvColumnIndex("Inputs"), iHid = csvColumnIndex("Hidden_Size"), iLay = csvColumnIndex("Layers"),
              iDen = csvColumnIndex("Agent_Density"), iCond = csvColumnIndex("Condition"), iTr = csvColumnIndex("Trial");
    for (size_t i = 1; i < lines.size(); ++i) {
        if (lines[i].empty()) continue;
        const auto fields = splitCsvLine(lines[i]);
        int in = 0, hid = 0, lay = 0, den = 0, tr = 0;
        Condition c{};
        if (fields.size() != expected || !parseIntField(fields[iIn], in) || !parseIntField(fields[iHid], hid) ||
            !parseIntField(fields[iLay], lay) || !parseIntField(fields[iDen], den) || !parseIntField(fields[iTr], tr) ||
            !parseCondition(fields[iCond], c)) {
            ++st.ignoredLines;
            continue;
        }
        st.completed.insert(resumeKey(in, hid, lay, den, c, tr));
        ++st.validRows;
    }
    return true;
}

// Builds one row with the same fixed/4-decimal formatting as the CSV stream; the row is then written with
// a single call so a row is never interleaved or half-formatted.
class CsvRow {
public:
    CsvRow() { os_ << std::fixed << std::setprecision(4); }
    CsvRow& text(const std::string& s) { sep(); os_ << s; return *this; }
    CsvRow& integer(long long v) { sep(); os_ << v; return *this; }
    CsvRow& num(double v) { sep(); if (std::isfinite(v)) os_ << v; return *this; } // NaN -> empty (N/A)
    CsvRow& blank() { sep(); return *this; }
    size_t fields() const { return n_; }
    std::string str() const { return os_.str(); }
private:
    void sep() { if (n_++) os_ << ','; }
    std::ostringstream os_;
    size_t n_ = 0;
};

// -----------------------------------------------------------------------------------------------------
// Test plan
// -----------------------------------------------------------------------------------------------------
struct TestCase {
    int inputs = 0, hidden = 0, layers = 0, density = 0, trial = 0;
    Condition cond = Condition::BASELINE;
    int block = 0;    // index of the block in the shuffled block order
    int blockPos = 0; // 0-based position of this condition inside its block
};

std::string modelFileName(const BenchConfig& cfg, const TestCase& tc) {
    return (fs::path(cfg.modelDir) / ("pathing_model_in" + std::to_string(tc.inputs) + "_H" + std::to_string(tc.hidden) +
        "_L" + std::to_string(tc.layers) + ".onnx")).string();
}

// -----------------------------------------------------------------------------------------------------
// One test
// -----------------------------------------------------------------------------------------------------
struct TestRow {
    std::string tsStart, tsEnd;
    DWORD_PTR aiMask = 0;
    std::string batchMode = "none";
    long long loopCount = 0;
    double preTestMs = 0.0, driftRefMs = 0.0, windowS = 0.0;
    bool driftFlag = false, truncated = false;
    SampleStats game, ai;
    long long missed = 0;
    double duty = 0.0, checksum = 0.0, agentsPerSec = 0.0;
    TelemetryAccum tel;
    // Validation columns: start delay (batch start minus release), share of batch time the thread was on a CPU,
    // batches per starting LP ("12:301 13:99") and batches that ended on another LP than they started on.
    SampleStats startDelay;
    double onCpuShare = std::numeric_limits<double>::quiet_NaN();
    std::string startLPs;
    long long lpChanged = 0;
    // Trace data, relative to the window start (written by main when --trace is set).
    long long windowStartTicks = 0;
    FrameTrace frames;
    std::vector<BatchTrace> batches;
};

// Thread cycles per second for AI_OnCPU_Share. QueryThreadCycleTime counts time-stamp-counter cycles on
// current x64 Windows, so the TSC rate, measured once at startup against QPC, converts them to seconds.
double g_tscHz = std::numeric_limits<double>::quiet_NaN();

double measureTscHz() {
    LARGE_INTEGER f{}, q0{}, q1{};
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&q0);
    const unsigned long long r0 = __rdtsc();
    Sleep(250);
    QueryPerformanceCounter(&q1);
    const unsigned long long r1 = __rdtsc();
    const double s = static_cast<double>(q1.QuadPart - q0.QuadPart) / static_cast<double>(f.QuadPart);
    return s > 0.0 ? static_cast<double>(r1 - r0) / s : std::numeric_limits<double>::quiet_NaN();
}

enum class TestStatus { Written, Skipped, Aborted };

TestStatus runTest(const TestCase& tc, const BenchConfig& cfg, Ort::Env& env, const CalibrationInfo& cal,
                   std::vector<float>& mainBuffer, TestRow& row, std::string& message, std::string& warning) {
    const bool hasAI = hasAIThread(tc.cond);
    const std::string modelPath = modelFileName(cfg, tc);
    if (hasAI) {
        std::error_code ec;
        if (!fs::is_regular_file(modelPath, ec)) {
            message = "model file not found: " + modelPath;
            return TestStatus::Skipped;
        }
    }

    // 1. Worker set for this condition (before the drift check).
    applyWorkerSet(workerSetFor(tc.cond));

    // 2. Pre-test drift check: 60 frames on LP0 under the condition's worker load.
    row.loopCount = g_loopCount.load();
    row.preTestMs = preTestDriftCheck(mainBuffer);
    row.driftRefMs = (parksSibling(tc.cond) && std::isfinite(cal.siblingParkedRefMs)) ? cal.siblingParkedRefMs : TARGET_FRAME_MS;
    row.driftFlag = std::abs(row.preTestMs - row.driftRefMs) / row.driftRefMs > DRIFT_TOLERANCE;
    std::cout << "  Pre-test frame (median of " << DRIFT_FRAMES << "): " << row.preTestMs << " ms (reference "
        << row.driftRefMs << " ms)" << (row.driftFlag ? "  ** DRIFT_FLAG **" : "") << "\n";

    // 3. Launch telemetry (opens PDH during warm-up), game and AI threads.
    TestSync sync;
    TelemetryControl tctl;
    std::vector<double> gameFrames;
    AIResult ai;
    const size_t gameCapacity = static_cast<size_t>(cfg.maxTestSeconds * 1000.0 / (TARGET_FRAME_MS / 4.0)) + 16;

    std::thread telemetry(telemetryThreadFunc, &tctl, &row.tel, cfg.telemetryIntervalMs);
    std::thread game(gameThreadFunc, &sync, &gameFrames, &row.frames, gameCapacity);
    std::thread aiThread;
    row.aiMask = aiMaskFor(tc.cond);
    if (hasAI) {
        AIParams p;
        p.env = &env;
        p.modelPath = fs::path(modelPath).wstring();
        p.inputs = tc.inputs;
        p.agents = tc.density;
        p.mask = row.aiMask;
        p.pin = tc.cond != Condition::UNPINNED;
        p.tickMs = cfg.tickMs;
        p.maxBatches = cfg.maxAIBatches;
        aiThread = std::thread(aiThreadFunc, p, &sync, &ai);
    }

    // 4. Warm-up (not measured): model load + AI warm-up batches while the game thread keeps LP0 busy.
    bool aborted = false, failed = false;
    const auto warmStart = Clock::now();
    for (;;) {
        if (sync.aiFailed.load(std::memory_order_acquire)) { failed = true; break; }
        const bool ready = sync.gameWarm.load(std::memory_order_acquire) &&
            (!hasAI || sync.aiWarm.load(std::memory_order_acquire)) && tctl.ready.load(std::memory_order_acquire);
        if (ready) break;
        if (g_abortRequested.load()) { aborted = true; break; }
        if (secondsSince(warmStart) > cfg.maxTestSeconds) { failed = true; message = "warm-up exceeded the wall-time cap"; break; }
        Sleep(POLL_MS);
    }

    // 5. Synchronized measurement window.
    FILETIME windowStartUtc{};
    Clock::time_point measureStart{};
    const double minBaselineWindowS = cfg.minAIBatches * cfg.tickMs / 1000.0;
    if (!aborted && !failed) {
        windowStartUtc = nowUtcFileTime();
        measureStart = Clock::now();
        sync.measuring.store(true, std::memory_order_release);
        sync.measuring.notify_all();
        tctl.begin.store(true, std::memory_order_release);

        while (!sync.stop.load(std::memory_order_acquire)) {
            if (g_abortRequested.load()) { aborted = true; requestStop(sync); break; }
            if (sync.aiFailed.load(std::memory_order_acquire)) { failed = true; requestStop(sync); break; }
            const double elapsed = secondsSince(measureStart);
            const bool gameDone = sync.gameFrames.load(std::memory_order_acquire) >= cfg.minGameFrames;
            // No-AI controls (BASELINE, SMT_IDLE): the window must also span the minimum AI window (minAIBatches ticks).
            const bool aiDone = hasAI ? sync.aiBatches.load(std::memory_order_acquire) >= cfg.minAIBatches
                                      : elapsed >= minBaselineWindowS;
            const bool lengthDone = elapsed >= cfg.minWindowSeconds; // --min-window-seconds (0 = no minimum)
            if ((gameDone && aiDone && lengthDone) || elapsed >= cfg.maxTestSeconds) { requestStop(sync); break; }
            Sleep(POLL_MS);
        }
    }
    else {
        requestStop(sync);
    }
    tctl.stop.store(true, std::memory_order_release);
    releaseWaiters(sync);

    game.join();
    if (aiThread.joinable()) aiThread.join();
    telemetry.join();

    if (sync.aiFailed.load()) failed = true;
    if (!ai.error.empty()) message = ai.error;
    if (!ai.warning.empty()) warning = ai.warning;
    if (aborted) { message = "aborted by user"; return TestStatus::Aborted; }
    if (failed) return TestStatus::Skipped;
    if (gameFrames.empty() || (hasAI && ai.batchMs.empty())) {
        message = "empty sample vector (game frames " + std::to_string(gameFrames.size()) + ", AI batches " +
            std::to_string(ai.batchMs.size()) + ")";
        return TestStatus::Skipped;
    }

    // 6. Statistics.
    const long long stopTicks = sync.stopTicks.load();
    const double windowMs = std::chrono::duration<double, std::milli>(Clock::duration(stopTicks) - measureStart.time_since_epoch()).count();
    row.windowS = windowMs / 1000.0;
    row.tsStart = isoLocal(windowStartUtc);
    row.tsEnd = isoLocal(fileTimeAddMs(windowStartUtc, windowMs));
    row.game = computeStats(gameFrames);
    row.truncated = static_cast<int>(gameFrames.size()) < cfg.minGameFrames ||
        (hasAI ? static_cast<int>(ai.batchMs.size()) < cfg.minAIBatches : row.windowS < minBaselineWindowS) ||
        row.windowS < cfg.minWindowSeconds;
    row.windowStartTicks = measureStart.time_since_epoch().count();
    if (hasAI) {
        row.batchMode = ai.batchMode;
        row.ai = computeStats(ai.batchMs);
        row.missed = ai.missedDeadlines;
        row.duty = windowMs > 0.0 ? ai.busyMs / windowMs : std::numeric_limits<double>::quiet_NaN();
        row.agentsPerSec = tc.density / row.ai.mean * 1000.0;
        // Mean over measured batches and agents of the sum of the agent's output values. A liveness /
        // consistency checksum that keeps the outputs consumed -- not a physical quantity.
        row.checksum = ai.outputSum / (static_cast<double>(ai.batchMs.size()) * tc.density);

        std::vector<double> delays;
        delays.reserve(ai.trace.size());
        unsigned long long cycles = 0;
        std::vector<long long> perLP(64, 0);
        for (const BatchTrace& b : ai.trace) {
            delays.push_back(std::chrono::duration<double, std::milli>(Clock::duration(b.start - b.release)).count());
            cycles += b.cycles;
            if (b.lpStart >= 0 && b.lpStart < 64) ++perLP[static_cast<size_t>(b.lpStart)];
            if (b.lpEnd != b.lpStart) ++row.lpChanged;
        }
        row.startDelay = computeStats(delays);
        if (ai.busyMs > 0.0 && std::isfinite(g_tscHz)) row.onCpuShare = static_cast<double>(cycles) / g_tscHz / (ai.busyMs / 1000.0);
        for (int lp = 0; lp < 64; ++lp) {
            if (perLP[static_cast<size_t>(lp)] == 0) continue;
            if (!row.startLPs.empty()) row.startLPs += " ";
            row.startLPs += std::to_string(lp) + ":" + std::to_string(perLP[static_cast<size_t>(lp)]);
        }
        row.batches = std::move(ai.trace);
    }
    return TestStatus::Written;
}

// Appends this test's trace rows. Times are written in ns relative to the start of the measurement window.
void writeTraces(std::ofstream& framesOut, std::ofstream& batchesOut, const std::string& traceId, const TestRow& r) {
    const long long t0 = r.windowStartTicks;
    auto ns = [t0](long long ticks) { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::duration(ticks - t0)).count(); };
    std::string buf;
    buf.reserve(r.frames.start.size() * 48);
    for (size_t i = 0; i < r.frames.start.size(); ++i)
        buf += traceId + "," + std::to_string(i) + "," + std::to_string(ns(r.frames.start[i])) + "," + std::to_string(ns(r.frames.end[i])) + "\n";
    framesOut << buf;
    framesOut.flush();
    buf.clear();
    for (size_t i = 0; i < r.batches.size(); ++i) {
        const BatchTrace& b = r.batches[i];
        buf += traceId + "," + std::to_string(i) + "," + std::to_string(ns(b.release)) + "," + std::to_string(ns(b.start)) + "," +
            std::to_string(ns(b.end)) + "," + (b.missed ? "1" : "0") + "," + std::to_string(b.lpStart) + "," +
            std::to_string(b.lpEnd) + "," + std::to_string(b.cycles) + "\n";
    }
    batchesOut << buf;
    batchesOut.flush();
}

std::string buildCsvRow(const std::string& runId, const std::string& traceId, const TestCase& tc, const TestRow& r) {
    const bool hasAI = hasAIThread(tc.cond);
    const double na = std::numeric_limits<double>::quiet_NaN();
    CsvRow row;
    row.text(runId).text(r.tsStart).text(r.tsEnd)
        .integer(tc.inputs).integer(tc.hidden).integer(tc.layers).integer(tc.density)
        .text(conditionName(tc.cond)).integer(tc.trial).integer(tc.blockPos)
        .text(hexMask(r.aiMask)).text(hasAI ? r.batchMode : "none").integer(r.loopCount)
        .num(r.preTestMs).num(r.driftRefMs).integer(r.driftFlag ? 1 : 0).integer(r.truncated ? 1 : 0).num(r.windowS)
        .integer(static_cast<long long>(r.game.n)).num(r.game.mean).num(r.game.median).num(r.game.min)
        .num(r.game.p99).num(r.game.p999).num(r.game.max).num(r.game.stddev)
        .num(r.game.p99 > 0.0 ? 1000.0 / r.game.p99 : na);
    if (hasAI) {
        row.integer(static_cast<long long>(r.ai.n)).num(r.ai.mean).num(r.ai.median).num(r.ai.p99).num(r.ai.max)
            .num(r.agentsPerSec).integer(r.missed).num(r.duty).num(r.checksum);
    }
    else {
        for (int i = 0; i < 9; ++i) row.blank(); // AI metrics are not applicable to the no-AI controls
    }
    const TelemetryAccum& t = r.tel;
    row.integer(t.samples)
        .num(t.thermalC.peakOr(-1)).num(t.thermalC.meanOr(-1))
        .num(t.powerMeterW.peakOr(-1)).num(t.powerMeterW.meanOr(-1))
        .num(t.raplPkgW.peakOr(-1)).num(t.raplPkgW.meanOr(-1))
        .num(t.perfP.peakOr(-1)).num(t.perfP.meanOr(-1))
        .num(t.perfE.peakOr(-1)).num(t.perfE.meanOr(-1))
        .num(t.pageFaults.peakOr(-1)).num(t.fileCacheFaults.peakOr(-1));
    row.text(traceId);
    if (hasAI) {
        row.num(r.startDelay.median).num(r.startDelay.p99).num(r.startDelay.max).num(r.onCpuShare)
            .text(r.startLPs).integer(r.lpChanged);
    }
    else {
        for (int i = 0; i < 6; ++i) row.blank();
    }
    if (row.fields() != csvColumns().size())
        fatal("internal error: CSV row has " + std::to_string(row.fields()) + " fields, header has " + std::to_string(csvColumns().size()));
    return row.str();
}

// -----------------------------------------------------------------------------------------------------
// System metadata
// -----------------------------------------------------------------------------------------------------
std::string cpuBrandString() {
    int regs[4] = {};
    __cpuid(regs, static_cast<int>(0x80000000));
    if (static_cast<unsigned>(regs[0]) < 0x80000004u) return "unknown";
    char brand[49] = {};
    for (int i = 0; i < 3; ++i) {
        __cpuid(regs, static_cast<int>(0x80000002 + i));
        std::memcpy(brand + 16 * i, regs, 16);
    }
    return trimCopy(brand);
}

std::string regString(HKEY root, const wchar_t* subKey, const wchar_t* name) {
    DWORD bytes = 0;
    if (RegGetValueW(root, subKey, name, RRF_RT_REG_SZ, nullptr, nullptr, &bytes) != ERROR_SUCCESS || bytes == 0) return "";
    std::wstring buf(bytes / sizeof(wchar_t) + 1, L'\0');
    bytes = static_cast<DWORD>(buf.size() * sizeof(wchar_t));
    if (RegGetValueW(root, subKey, name, RRF_RT_REG_SZ, nullptr, buf.data(), &bytes) != ERROR_SUCCESS) return "";
    buf.resize(wcslen(buf.c_str()));
    return narrow(buf);
}

bool regDword(HKEY root, const wchar_t* subKey, const wchar_t* name, DWORD& value) {
    DWORD bytes = sizeof(value);
    return RegGetValueW(root, subKey, name, RRF_RT_REG_DWORD, nullptr, &value, &bytes) == ERROR_SUCCESS;
}

std::string windowsVersionString() {
    RTL_OSVERSIONINFOW vi{};
    vi.dwOSVersionInfoSize = sizeof(vi);
    using RtlGetVersionFn = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);
    std::string s = "unknown";
    if (HMODULE ntdll = GetModuleHandleW(L"ntdll.dll")) {
        auto fn = reinterpret_cast<RtlGetVersionFn>(reinterpret_cast<void*>(GetProcAddress(ntdll, "RtlGetVersion")));
        if (fn && fn(&vi) == 0) {
            s = std::to_string(vi.dwMajorVersion) + "." + std::to_string(vi.dwMinorVersion) + "." + std::to_string(vi.dwBuildNumber);
        }
    }
    const wchar_t* cv = L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion";
    DWORD ubr = 0;
    if (regDword(HKEY_LOCAL_MACHINE, cv, L"UBR", ubr)) s += "." + std::to_string(ubr);
    const std::string display = regString(HKEY_LOCAL_MACHINE, cv, L"DisplayVersion");
    if (!display.empty()) s += " (" + display + ")";
    return s;
}

// System model and firmware as Windows reports them (SMBIOS values copied to the registry at boot).
std::string biosString() {
    const wchar_t* key = L"HARDWARE\\DESCRIPTION\\System\\BIOS";
    auto v = [&](const wchar_t* name) {
        const std::string s = regString(HKEY_LOCAL_MACHINE, key, name);
        return s.empty() ? std::string("unknown") : s;
    };
    return v(L"SystemManufacturer") + " " + v(L"SystemProductName") + " | board " + v(L"BaseBoardProduct") +
        " | BIOS " + v(L"BIOSVersion") + " (" + v(L"BIOSReleaseDate") + ")";
}

std::string guidToString(const GUID& g) {
    char b[64];
    snprintf(b, sizeof(b), "%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        static_cast<unsigned long>(g.Data1), static_cast<unsigned>(g.Data2), static_cast<unsigned>(g.Data3),
        g.Data4[0], g.Data4[1], g.Data4[2], g.Data4[3], g.Data4[4], g.Data4[5], g.Data4[6], g.Data4[7]);
    return b;
}

std::string activePowerSchemeString() {
    GUID* scheme = nullptr;
    if (PowerGetActiveScheme(nullptr, &scheme) != ERROR_SUCCESS || !scheme) return "unknown";
    std::string s = guidToString(*scheme);
    DWORD size = 0;
    const DWORD rc = PowerReadFriendlyName(nullptr, scheme, nullptr, nullptr, nullptr, &size);
    if ((rc == ERROR_SUCCESS || rc == ERROR_MORE_DATA) && size > 0) {
        std::vector<UCHAR> nameBuf(size + sizeof(wchar_t), 0);
        if (PowerReadFriendlyName(nullptr, scheme, nullptr, nullptr, nameBuf.data(), &size) == ERROR_SUCCESS)
            s += " (" + narrow(reinterpret_cast<const wchar_t*>(nameBuf.data())) + ")";
    }
    LocalFree(scheme);
    return s;
}

std::string acStatusString(bool* online) {
    SYSTEM_POWER_STATUS sps{};
    if (!GetSystemPowerStatus(&sps)) { if (online) *online = false; return "unknown (GetSystemPowerStatus failed)"; }
    if (online) *online = (sps.ACLineStatus == 1);
    std::string s = sps.ACLineStatus == 1 ? "online" : (sps.ACLineStatus == 0 ? "OFFLINE (battery)" : "unknown");
    s += " (ACLineStatus=" + std::to_string(static_cast<int>(sps.ACLineStatus));
    if (sps.BatteryLifePercent != 255) s += ", battery " + std::to_string(static_cast<int>(sps.BatteryLifePercent)) + "%";
    s += ")";
    return s;
}

std::string exePath() {
    wchar_t buf[MAX_PATH * 4] = {};
    const DWORD n = GetModuleFileNameW(nullptr, buf, static_cast<DWORD>(std::size(buf)));
    return n ? narrow(std::wstring(buf, n)) : "unknown";
}

void appendMetadata(const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::app | std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[WARN] could not append to " << path << "\n";
        return;
    }
    f << text;
}

// -----------------------------------------------------------------------------------------------------
// Command line / console control
// -----------------------------------------------------------------------------------------------------
BOOL WINAPI consoleCtrlHandler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT) {
        if (g_abortRequested.exchange(true)) return FALSE; // second Ctrl+C: default handling (terminate)
        return TRUE;
    }
    return FALSE;
}

// -----------------------------------------------------------------------------------------------------
// Output verification (--verify-outputs DIR). A numerical check outside the timed path: every grid model runs
// in this build with makeSessionOptions(), the same tensor layout and the same probe and warm-up runs as a
// timed test, and the outputs of the first measured batch are written for comparison with PyTorch
// (verify_cpp_outputs.py). No process setup, soak or timing.
// -----------------------------------------------------------------------------------------------------
// Hash-based inputs in [lo, hi); verify_cpp_outputs.py reproduces them bit for bit.
float hashUniform(uint32_t index, uint32_t seed, float lo, float hi) {
    uint32_t h = index * 0x9E3779B1u + seed * 0x85EBCA77u;
    h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
    const float u = static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
    return lo + (hi - lo) * u;
}

int verifyOutputs(const BenchConfig& cfg, const std::vector<int>& inputSizes, const std::vector<int>& hiddenSizes,
                  const std::vector<int>& layerCounts, const std::vector<int>& agentDensities) {
    // Topology is not set up in this mode, so the thread is pinned directly rather than with pinCurrentThreadOrDie.
    if (cfg.verifyLP >= 0 && SetThreadAffinityMask(GetCurrentThread(), lpBit(cfg.verifyLP)) == 0) {
        std::cerr << "[ERROR] cannot pin the verification to LP " << cfg.verifyLP << " (GetLastError=" << GetLastError() << ")\n";
        return 1;
    }
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON); // same floating-point mode as the AI thread
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    struct InputSet { const char* name; uint32_t seed; float lo, hi; };
    const InputSet sets[] = { { "bench", 0, 0.0f, 0.0f }, { "u1_s1", 1, -1.0f, 1.0f }, { "u1_s2", 2, -1.0f, 1.0f },
                              { "u10_s3", 3, -10.0f, 10.0f } };

    const fs::path dir(cfg.verifyDir);
    fs::create_directories(dir);
    std::ofstream man(dir / "verify_manifest.csv");
    if (!man) { std::cerr << "[ERROR] cannot write " << (dir / "verify_manifest.csv").string() << "\n"; return 1; }
    man << "inputs,hidden,layers,agents,input_set,seed,lo,hi,input_bits_sum,outputs,all_finite,repeat_identical,file\n";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "AI_Pathing_verify");
    const std::string ortVersion = OrtGetApiBase()->GetVersionString();
    std::cout << "Output verification: ONNX Runtime " << ortVersion << ", LP "
              << (cfg.verifyLP >= 0 ? std::to_string(cfg.verifyLP) : std::string("not pinned")) << ", models from "
              << cfg.modelDir << " -> " << dir.string() << "\n";
    {
        std::ofstream info(dir / "verify_info.txt");
        info << "onnxruntime " << ortVersion << "\nlp " << cfg.verifyLP << "\nstarted " << isoLocalNow() << "\n";
    }

    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int problems = 0;
    for (int in : inputSizes) for (int hid : hiddenSizes) for (int lay : layerCounts) {
        TestCase tc;
        tc.inputs = in; tc.hidden = hid; tc.layers = lay;
        const fs::path model = modelFileName(cfg, tc);
        for (int agents : agentDensities) {
            // A new session per (model, agents), as in a timed test.
            Ort::Session session(env, model.wstring().c_str(), makeSessionOptions());
            Ort::AllocatorWithDefaultOptions allocator;
            const std::string inName = session.GetInputNameAllocated(0, allocator).get();
            const std::string outName = session.GetOutputNameAllocated(0, allocator).get();
            const char* inNames[] = { inName.c_str() };
            const char* outNames[] = { outName.c_str() };
            const std::vector<int64_t> shape = { agents, SEQ_LEN, in };
            std::vector<float> buffer(static_cast<size_t>(agents) * SEQ_LEN * in);
            for (const InputSet& s : sets) {
                if (s.seed == 0) fillBenchmarkInputs(buffer);
                else for (size_t i = 0; i < buffer.size(); ++i) buffer[i] = hashUniform(static_cast<uint32_t>(i), s.seed, s.lo, s.hi);
                uint64_t bitsSum = 0;
                for (float v : buffer) bitsSum += std::bit_cast<uint32_t>(v);

                Ort::Value tensor = Ort::Value::CreateTensor<float>(memInfo, buffer.data(), buffer.size(), shape.data(), shape.size());
                auto run = [&]() {
                    auto outs = session.Run(Ort::RunOptions{ nullptr }, inNames, &tensor, 1, outNames, 1);
                    const size_t n = outs.front().GetTensorTypeAndShapeInfo().GetElementCount();
                    const float* d = outs.front().GetTensorData<float>();
                    return std::vector<float>(d, d + n);
                };
                const std::vector<float> first = run();            // the probe run of a timed test
                for (int w = 0; w < AI_WARMUP_BATCHES; ++w) run(); // its warm-up batches
                const std::vector<float> out = run();              // its first measured batch
                const bool finite = std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v); });
                const bool repeat = first.size() == out.size() &&
                    std::memcmp(first.data(), out.data(), out.size() * sizeof(float)) == 0;
                if (!finite || !repeat || out.size() != static_cast<size_t>(agents) * 3) ++problems;

                const std::string file = "in" + std::to_string(in) + "_H" + std::to_string(hid) + "_L" + std::to_string(lay) +
                    "_A" + std::to_string(agents) + "_" + s.name + ".f32";
                std::ofstream bin(dir / file, std::ios::binary);
                bin.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
                man << in << ',' << hid << ',' << lay << ',' << agents << ',' << s.name << ',' << s.seed << ',' << s.lo << ','
                    << s.hi << ',' << bitsSum << ',' << out.size() << ',' << (finite ? 1 : 0) << ',' << (repeat ? 1 : 0) << ','
                    << file << '\n';
            }
        }
        std::cout << "  in" << in << "_H" << hid << "_L" << lay << " done\n";
    }
    std::cout << "Output verification finished: " << problems << " case(s) with non-finite, non-repeatable or mis-sized outputs.\n";
    return problems == 0 ? 0 : 1;
}

void printUsage() {
    BenchConfig d;
    std::cout <<
        "Usage: LocalSequenceGameOptimizationCores.exe [options]\n"
        "  --allow-battery          run even when not on AC power\n"
        "  --recalibrate            ignore calibrated_loops.txt and calibrate again after the soak\n"
        "  --soak-seconds N         full-load soak before calibration/tests (default " << d.soakSeconds << ")\n"
        "  --trials N               trials per configuration (default " << d.trials << ")\n"
        "  --tick-ms X              AI tick period in ms (default " << d.tickMs << ")\n"
        "  --min-game-frames N      (default " << d.minGameFrames << ")\n"
        "  --min-ai-batches N       (default " << d.minAIBatches << ")\n"
        "  --max-ai-batches N       (default " << d.maxAIBatches << ")\n"
        "  --max-test-seconds X     wall-time cap per measurement window (default " << d.maxTestSeconds << ")\n"
        "  --seed N                 block-randomization seed (default " << d.seed << ")\n"
        "  --max-tests N            stop after N written tests this session (default: no limit)\n"
        "  --telemetry-ms N         PDH sampling interval (default " << d.telemetryIntervalMs << ")\n"
        "  --csv FILE               results CSV (default " << d.csvPath << ")\n"
        "  --model-dir DIR          ONNX model folder (default " << d.modelDir << ")\n"
        "  --conditions LIST        comma-separated subset of BASELINE,P_SHARED,P_SMT,E_CORE,SMT_IDLE,\n"
        "                           E_CORE_SMT_IDLE,P_SHARED_SMT_IDLE,E_CORE_1LP,UNPINNED\n"
        "                           (default BASELINE,P_SHARED,P_SMT,E_CORE). Rows already in the CSV count as\n"
        "                           done, so use a new --csv file for a supplementary session.\n"
        "  --configs FILE           run only the configurations in FILE, one 'inputs,hidden,layers,agents' per line\n"
        "                           (# starts a comment); default: the full grid\n"
        "  --min-window-seconds X   every measurement window also lasts at least X s (default 0: no minimum)\n"
        "  --trace PREFIX           write PREFIX_batches.csv and PREFIX_frames.csv (one row per measured AI batch\n"
        "                           and game frame), linked to the results CSV by Trace_ID\n"
        "  --note TEXT              operator note (room temperature, MSI Center mode, ...) written to the metadata\n"
        "  --verify-outputs DIR     write every grid model's outputs (benchmark and hash inputs) to DIR and exit;\n"
        "                           no soak, calibration or timing (compare with verify_cpp_outputs.py)\n"
        "  --verify-lp N            pin the output verification to LP N (default: not pinned)\n";
}

bool parseArgs(int argc, char** argv, BenchConfig& cfg, std::string& err, bool& helpOnly) {
    helpOnly = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(a + " requires a value");
            return argv[++i];
        };
        try {
            if (a == "--help" || a == "-h") { helpOnly = true; return true; }
            else if (a == "--allow-battery") cfg.allowBattery = true;
            else if (a == "--recalibrate") cfg.recalibrate = true;
            else if (a == "--soak-seconds") cfg.soakSeconds = std::stoi(value());
            else if (a == "--trials") cfg.trials = std::stoi(value());
            else if (a == "--tick-ms") cfg.tickMs = std::stod(value());
            else if (a == "--min-game-frames") cfg.minGameFrames = std::stoi(value());
            else if (a == "--min-ai-batches") cfg.minAIBatches = std::stoi(value());
            else if (a == "--max-ai-batches") cfg.maxAIBatches = std::stoi(value());
            else if (a == "--max-test-seconds") cfg.maxTestSeconds = std::stod(value());
            else if (a == "--seed") cfg.seed = static_cast<uint32_t>(std::stoul(value()));
            else if (a == "--max-tests") cfg.maxTests = std::stoi(value());
            else if (a == "--telemetry-ms") cfg.telemetryIntervalMs = std::stoi(value());
            else if (a == "--csv") cfg.csvPath = value();
            else if (a == "--model-dir") cfg.modelDir = value();
            else if (a == "--conditions") cfg.conditions = value();
            else if (a == "--verify-outputs") cfg.verifyDir = value();
            else if (a == "--verify-lp") cfg.verifyLP = std::stoi(value());
            else if (a == "--configs") cfg.configsPath = value();
            else if (a == "--min-window-seconds") cfg.minWindowSeconds = std::stod(value());
            else if (a == "--trace") cfg.tracePrefix = value();
            else if (a == "--note") cfg.note = value();
            else { err = "unknown argument: " + a; return false; }
        }
        catch (const std::exception& e) {
            err = "invalid value for " + a + " (" + e.what() + ")";
            return false;
        }
    }
    if (cfg.soakSeconds < 0 || cfg.trials < 1 || cfg.tickMs <= 0.0 || cfg.minGameFrames < 1 || cfg.minAIBatches < 1 ||
        cfg.maxAIBatches < cfg.minAIBatches || cfg.maxTestSeconds <= 0.0 || cfg.telemetryIntervalMs < 10 ||
        cfg.minWindowSeconds < 0.0 || cfg.minWindowSeconds >= cfg.maxTestSeconds) {
        err = "invalid option combination (check ranges: trials>=1, tick>0, max-ai-batches>=min-ai-batches, telemetry-ms>=10, "
              "0<=min-window-seconds<max-test-seconds)";
        return false;
    }
    return true;
}

// --configs file: one "inputs,hidden,layers,agents" per line; blank lines and text after '#' are ignored.
struct GridConfig { int inputs, hidden, layers, density; };

bool loadConfigList(const std::string& path, std::vector<GridConfig>& out, std::string& err) {
    std::ifstream f(path);
    if (!f.is_open()) { err = "cannot open --configs file '" + path + "'"; return false; }
    std::string line;
    int lineNo = 0;
    while (std::getline(f, line)) {
        ++lineNo;
        const auto hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        line = trimCopy(line);
        if (line.empty()) continue;
        const auto parts = splitCsvLine(line);
        GridConfig g{};
        if (parts.size() != 4 || !parseIntField(trimCopy(parts[0]), g.inputs) || !parseIntField(trimCopy(parts[1]), g.hidden) ||
            !parseIntField(trimCopy(parts[2]), g.layers) || !parseIntField(trimCopy(parts[3]), g.density) ||
            g.inputs < 1 || g.hidden < 1 || g.layers < 1 || g.density < 1) {
            err = path + " line " + std::to_string(lineNo) + ": expected 'inputs,hidden,layers,agents', got '" + line + "'";
            return false;
        }
        for (const GridConfig& o : out) {
            if (o.inputs == g.inputs && o.hidden == g.hidden && o.layers == g.layers && o.density == g.density) {
                err = path + " line " + std::to_string(lineNo) + ": duplicate configuration '" + line + "'";
                return false;
            }
        }
        out.push_back(g);
    }
    if (out.empty()) { err = "--configs file '" + path + "' lists no configurations"; return false; }
    return true;
}

// =====================================================================================================
// main
// =====================================================================================================
int main(int argc, char** argv) {
    std::cout << std::unitbuf << std::fixed << std::setprecision(2);
    std::cout << "=== Steady-State Hybrid-Core AI Placement Benchmark ===\n\n";

    BenchConfig cfg;
    std::string err;
    bool helpOnly = false;
    if (!parseArgs(argc, argv, cfg, err, helpOnly)) {
        std::cerr << "[ERROR] " << err << "\n";
        printUsage();
        return 2;
    }
    if (helpOnly) { printUsage(); return 0; }

    // ---- Placement conditions (edit this list; default = BASELINE, P_SHARED, P_SMT, E_CORE) -------------
    std::vector<Condition> conditions = { Condition::BASELINE, Condition::P_SHARED, Condition::P_SMT, Condition::E_CORE };
    if (!cfg.conditions.empty()) {
        conditions.clear();
        for (const std::string& part : splitCsvLine(cfg.conditions)) {
            const std::string name = trimCopy(part);
            Condition c{};
            if (!parseCondition(name, c)) {
                std::string valid;
                for (Condition v : ALL_CONDITIONS) valid += (valid.empty() ? "" : ", ") + std::string(conditionName(v));
                std::cerr << "[ERROR] unknown condition '" << name << "' in --conditions (valid: " << valid << ")\n";
                printUsage();
                return 2;
            }
            conditions.push_back(c);
        }
    }

    // ---- Experiment grid ----------------------------------------------------------------------------------
    const std::vector<int> inputSizes = { 10, 100, 1000 };
    const std::vector<int> hiddenSizes = { 32, 64, 128, 256, 512 };
    const std::vector<int> layerCounts = { 1, 2, 4, 8};
    const std::vector<int> agentDensities = { 50, 100, 200 };

    if (!cfg.verifyDir.empty()) return verifyOutputs(cfg, inputSizes, hiddenSizes, layerCounts, agentDensities);

    // Configurations to run: the full grid, or the --configs list (whose model files must all exist).
    std::vector<GridConfig> configs;
    if (cfg.configsPath.empty()) {
        for (int in : inputSizes)
            for (int hid : hiddenSizes)
                for (int lay : layerCounts)
                    for (int den : agentDensities) configs.push_back({ in, hid, lay, den });
    }
    else {
        if (!loadConfigList(cfg.configsPath, configs, err)) {
            std::cerr << "[ERROR] " << err << "\n";
            return 2;
        }
        for (const GridConfig& g : configs) {
            TestCase tc;
            tc.inputs = g.inputs; tc.hidden = g.hidden; tc.layers = g.layers;
            std::error_code ec;
            if (!fs::is_regular_file(modelFileName(cfg, tc), ec)) {
                std::cerr << "[ERROR] --configs: model file not found: " << modelFileName(cfg, tc) << "\n";
                return 2;
            }
        }
    }

    const std::string runId = makeRunId();
    const std::string sessionStart = isoLocalNow();
    SetConsoleCtrlHandler(consoleCtrlHandler, TRUE);
    std::cout << "Run_ID " << runId << " | started " << sessionStart << "\n";

    // ---- 1. Process setup ---------------------------------------------------------------------------------
    std::string priorityStatus = "HIGH_PRIORITY_CLASS (ok)";
    if (!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)) {
        priorityStatus = "SetPriorityClass(HIGH_PRIORITY_CLASS) FAILED, GetLastError=" + std::to_string(GetLastError());
        std::cerr << "[WARN] " << priorityStatus << "\n";
    }
    PROCESS_POWER_THROTTLING_STATE throttling{};
    throttling.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    throttling.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
    throttling.StateMask = 0; // ControlMask set + StateMask clear = always OFF (no EcoQoS, honor timer resolution)
    std::string throttlingStatus = "EXECUTION_SPEED + IGNORE_TIMER_RESOLUTION throttling disabled (ok)";
    if (!SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &throttling, sizeof(throttling))) {
        throttlingStatus = "SetProcessInformation(ProcessPowerThrottling) FAILED, GetLastError=" + std::to_string(GetLastError());
        std::cerr << "[WARN] " << throttlingStatus << "\n";
    }
    bool acOnline = false;
    const std::string acStatus = acStatusString(&acOnline);
    std::cout << "AC power: " << acStatus << "\n";
    if (!acOnline && !cfg.allowBattery) {
        std::cerr << "[ERROR] Not running on AC power. Plug in the charger, or pass --allow-battery to override.\n";
        return 1;
    }

    // ---- 2. Topology --------------------------------------------------------------------------------------
    if (!setupTopology(err)) {
        std::cerr << "[ERROR] Topology setup failed: " << err << "\n";
        return 1;
    }
    printTopology();

    std::vector<Condition> active;
    for (Condition c : conditions) {
        if (parksSibling(c) && g_topo.siblingLP < 0) {
            std::cerr << "[WARN] " << conditionName(c) << " removed: LP " << g_topo.gameLP << " has no SMT sibling (SMT disabled?).\n";
            continue;
        }
        if (needsECores(c) && !g_topo.hybrid) {
            std::cerr << "[WARN] " << conditionName(c) << " removed: CPU is not hybrid (single EfficiencyClass).\n";
            continue;
        }
        if (std::find(active.begin(), active.end(), c) == active.end()) active.push_back(c);
    }
    conditions = active;
    if (conditions.empty()) {
        std::cerr << "[ERROR] No runnable placement conditions.\n";
        return 1;
    }
    std::string conditionList;
    for (Condition c : conditions) conditionList += (conditionList.empty() ? "" : ",") + std::string(conditionName(c));

    if (g_topo.housekeepingLP >= 0) pinCurrentThreadOrDie(lpBit(g_topo.housekeepingLP), "main thread");

    // AI tick pacing timer.
    if (HANDLE probe = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS)) {
        g_highResTimerAvailable = true;
        CloseHandle(probe);
    }
    else {
        g_highResTimerAvailable = false;
        timeBeginPeriod(1);
        std::cerr << "[WARN] High-resolution waitable timer unavailable; AI pacing falls back to timeBeginPeriod(1)+Sleep.\n";
    }
    const std::string pacingTimer = g_highResTimerAvailable ? "high-resolution waitable timer (CREATE_WAITABLE_TIMER_HIGH_RESOLUTION)"
                                                            : "timeBeginPeriod(1) + Sleep fallback";

    // Telemetry availability.
    const TelemetryProbe tprobe = probeTelemetry();
    for (const auto& p : tprobe.problems) std::cerr << "[WARN] Telemetry: " << p << "\n";
    std::cout << "Telemetry: ACPI thermal=" << tprobe.thermal << " | PowerMeter=" << (tprobe.powerMeter ? "yes" : "no")
        << " | RAPL pkg=" << (tprobe.rapl ? "yes" : "no") << " | ProcPerf=" << procPerfMethodName(tprobe.procPerfMethod) << "\n";

    // Time-stamp-counter rate for AI_OnCPU_Share.
    g_tscHz = measureTscHz();

    // ONNX Runtime: one environment for the whole process, shared by every AI thread.
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "AI_Pathing");
    const std::string ortVersion = OrtGetApiBase()->GetVersionString();
    std::cout << "ONNX Runtime " << ortVersion << "\n\n";

    // ---- Resume state + block-randomized plan ------------------------------------------------------------
    ResumeState resume;
    if (!loadResumeState(cfg.csvPath, resume, err)) {
        std::cerr << "[ERROR] " << err << "\n";
        return 1;
    }
    if (resume.hasContent) {
        std::cout << "[Resume] " << cfg.csvPath << ": " << resume.validRows << " complete rows";
        if (resume.ignoredLines) std::cout << ", " << resume.ignoredLines << " partial/malformed lines ignored";
        std::cout << "\n";
    }

    struct Block { int inputs, hidden, layers, density, trial; };
    std::vector<Block> blocks;
    for (const GridConfig& g : configs)
        for (int t = 1; t <= cfg.trials; ++t) blocks.push_back({ g.inputs, g.hidden, g.layers, g.density, t });

    std::mt19937 rng(cfg.seed);
    deterministicShuffle(blocks, rng);
    std::vector<TestCase> plan;
    plan.reserve(blocks.size() * conditions.size());
    for (size_t b = 0; b < blocks.size(); ++b) {
        std::vector<Condition> order = conditions;
        deterministicShuffle(order, rng);
        for (size_t pos = 0; pos < order.size(); ++pos) {
            TestCase tc;
            tc.inputs = blocks[b].inputs; tc.hidden = blocks[b].hidden; tc.layers = blocks[b].layers;
            tc.density = blocks[b].density; tc.trial = blocks[b].trial; tc.cond = order[pos];
            tc.block = static_cast<int>(b); tc.blockPos = static_cast<int>(pos);
            plan.push_back(tc);
        }
    }
    auto isDone = [&](const TestCase& tc) {
        return resume.completed.count(resumeKey(tc.inputs, tc.hidden, tc.layers, tc.density, tc.cond, tc.trial)) > 0;
    };
    const size_t remainingAtStart = static_cast<size_t>(std::count_if(plan.begin(), plan.end(), [&](const TestCase& tc) { return !isDone(tc); }));
    std::cout << "[Plan] " << plan.size() << " tests (" << blocks.size() << " blocks x " << conditions.size()
        << " conditions: " << conditionList << "), seed " << cfg.seed << ", " << plan.size() - remainingAtStart
        << " already complete, " << remainingAtStart << " remaining\n";
    if (remainingAtStart == 0) {
        std::cout << "Nothing to do.\n";
        return 0;
    }
    const double aPrioriPerTestS = std::max({ cfg.minGameFrames * TARGET_FRAME_MS / 1000.0, cfg.minAIBatches * cfg.tickMs / 1000.0,
        cfg.minWindowSeconds }) + 2.5;
    std::cout << "[Plan] a-priori lower-bound estimate: " << formatDuration(aPrioriPerTestS * remainingAtStart + cfg.soakSeconds)
        << " (P_SHARED and heavy models take longer)\n\n";

    // ---- Calibration file ---------------------------------------------------------------------------------
    CalibrationInfo cal;
    bool needCalibration = true;
    if (!cfg.recalibrate) {
        const CalLoad cl = loadCalibration(cfg.calibrationPath, cal);
        if (cl == CalLoad::Ok && cal.gameLP == g_topo.gameLP) {
            needCalibration = false;
            g_loopCount.store(cal.loopCount);
        }
        else if (cl == CalLoad::Ok) {
            std::cout << "[Calibration] Stored calibration was made on LP " << cal.gameLP << "; recalibrating for LP " << g_topo.gameLP << ".\n";
        }
        else if (cl == CalLoad::Legacy) {
            const std::string legacy = cfg.calibrationPath + ".legacy.txt";
            std::error_code ec;
            fs::rename(cfg.calibrationPath, legacy, ec);
            std::cout << "[Calibration] " << cfg.calibrationPath << " is a pre-soak (legacy) calibration; moved to " << legacy
                << " and recalibrating after the soak.\n";
        }
        else if (cl == CalLoad::Invalid) {
            std::cout << "[Calibration] " << cfg.calibrationPath << " is unreadable; recalibrating.\n";
        }
    }

    // ---- Metadata (system) --------------------------------------------------------------------------------
    {
        std::ostringstream m;
        std::string cmdLine;
        for (int i = 0; i < argc; ++i) cmdLine += (i ? " " : "") + std::string(argv[i]);
        std::string overlay = regString(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Control\\Power\\User\\PowerSchemes", L"ActiveOverlayAcPowerScheme");
        DWORD vbs = 0, hvci = 0;
        const bool hasVbs = regDword(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Control\\DeviceGuard", L"EnableVirtualizationBasedSecurity", vbs);
        const bool hasHvci = regDword(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Control\\DeviceGuard\\Scenarios\\HypervisorEnforcedCodeIntegrity", L"Enabled", hvci);
        std::error_code ec;
        auto list = [](const std::vector<int>& v) {
            std::string s;
            for (int x : v) s += (s.empty() ? "" : ",") + std::to_string(x);
            return "{" + s + "}";
        };
        m << "\n==================== Session " << runId << " ====================\n"
          << "session_start            : " << sessionStart << "\n"
          << "executable               : " << exePath() << "\n"
          << "working_directory        : " << fs::current_path(ec).string() << "\n"
          << "command_line             : " << cmdLine << "\n"
          << "cpu_brand                : " << cpuBrandString() << "\n"
          << "logical_processors       : " << g_topo.logicalCount << " (active processor groups: " << GetActiveProcessorGroupCount() << ")\n"
          << "hybrid                   : " << (g_topo.hybrid ? "yes" : "no") << " (P physical " << g_topo.pPhysical << ", E physical " << g_topo.ePhysical << ")\n"
          << "process_affinity_mask    : " << hexMask(g_topo.processMask) << "\n"
          << "p_core_mask              : " << hexMask(g_topo.pCoreMask) << " [" << lpList(g_topo.pCoreMask) << "]\n"
          << "e_core_mask              : " << hexMask(g_topo.eCoreMask) << " [" << lpList(g_topo.eCoreMask) << "]\n"
          << "game_lp (LP0)            : " << g_topo.gameLP << "\n"
          << "game_lp_smt_sibling      : " << g_topo.siblingLP << "\n"
          << "worker_mask              : " << hexMask(g_topo.workerMask) << " [" << lpList(g_topo.workerMask) << "]\n"
          << "housekeeping_lp          : " << g_topo.housekeepingLP << "\n"
          << "e_core_ai_mask           : " << hexMask(g_topo.eCoreAIMask) << " [" << lpList(g_topo.eCoreAIMask) << "]\n"
          << "windows_version          : " << windowsVersionString() << "\n"
          << "vbs_registry_config      : " << (hasVbs ? "EnableVirtualizationBasedSecurity=" + std::to_string(vbs) : std::string("not set")) << " (registry configuration, not runtime state)\n"
          << "hvci_registry_config     : " << (hasHvci ? "Enabled=" + std::to_string(hvci) : std::string("not set")) << " (registry configuration, not runtime state)\n"
          << "onnxruntime_version      : " << ortVersion << "\n"
          << "compiler                 : MSVC _MSC_FULL_VER=" << _MSC_FULL_VER << ", built " << __DATE__ << " " << __TIME__
#ifdef NDEBUG
          << " (NDEBUG)\n"
#else
          << " (DEBUG BUILD - not valid for measurements)\n"
#endif
          << "ac_power                 : " << acStatus << (cfg.allowBattery ? " [--allow-battery]" : "") << "\n"
          << "power_scheme             : " << activePowerSchemeString() << "\n"
          << "power_mode_overlay_ac    : " << (overlay.empty() ? std::string("not set") : overlay) << " (all-zero GUID = no overlay)\n"
          << "priority_class           : " << priorityStatus << "\n"
          << "power_throttling         : " << throttlingStatus << "\n"
          << "ai_pacing_timer          : " << pacingTimer << "\n"
          << "telemetry                : thermal=" << tprobe.thermal << "; power_meter=" << (tprobe.powerMeter ? "yes" : "no")
          << "; rapl_pkg=" << (tprobe.rapl ? "yes" : "no") << "; page_faults=" << (tprobe.pageFaults ? "yes" : "no")
          << "; file_cache_faults=" << (tprobe.fileCacheFaults ? "yes" : "no") << "; interval_ms=" << cfg.telemetryIntervalMs << "\n"
          << "procperf_method          : " << procPerfMethodName(tprobe.procPerfMethod) << "\n"
          << "rng_seed                 : " << cfg.seed << " (mt19937, implementation-independent Fisher-Yates)\n"
          << "conditions               : " << conditionList << "\n"
          << "config                   : trials=" << cfg.trials << " min_game_frames=" << cfg.minGameFrames
          << " min_ai_batches=" << cfg.minAIBatches << " max_ai_batches=" << cfg.maxAIBatches
          << " max_test_s=" << cfg.maxTestSeconds << " min_window_s=" << cfg.minWindowSeconds << " tick_ms=" << cfg.tickMs
          << " soak_s=" << cfg.soakSeconds << " drift_tolerance=" << DRIFT_TOLERANCE << " target_frame_ms=" << TARGET_FRAME_MS << "\n";
        if (cfg.configsPath.empty()) {
            m << "grid                     : inputs=" << list(inputSizes) << " hidden=" << list(hiddenSizes)
              << " layers=" << list(layerCounts) << " densities=" << list(agentDensities) << "\n";
        }
        else {
            m << "configs                  : " << configs.size() << " from " << cfg.configsPath << ":";
            for (const GridConfig& g : configs) m << " in" << g.inputs << "_H" << g.hidden << "_L" << g.layers << "_A" << g.density;
            m << "\n";
        }
        m << "plan                     : " << plan.size() << " tests, " << remainingAtStart << " remaining at session start\n"
          << "csv                      : " << cfg.csvPath << "\n"
          << "trace                    : " << (cfg.tracePrefix.empty() ? std::string("off") : cfg.tracePrefix + "_batches.csv, " + cfg.tracePrefix + "_frames.csv") << "\n"
          << "tsc_hz                   : " << std::fixed << std::setprecision(0) << g_tscHz << " (AI_OnCPU_Share = thread cycles / tsc_hz / batch time)\n"
          << "system_bios              : " << biosString() << "\n"
          << "operator_note            : " << (cfg.note.empty() ? std::string("none") : cfg.note) << "\n";
        for (const auto& p : tprobe.problems) m << "telemetry_warning        : " << p << "\n";
        appendMetadata(cfg.metadataPath, m.str());
    }

    // ---- 3. Persistent background workers -----------------------------------------------------------------
    startWorkers();
    std::cout << "[Workers] " << g_workers.size() << " persistent workers started on LPs " << lpList(g_topo.workerMask) << "\n";
    Sleep(200);

    int exitCode = 0;
    std::vector<float> mainBuffer(PHYSICS_BUFFER_ELEMENTS, 1.0f); // main thread's physics buffer (LP0 checks)

    // ---- 4. Soak, then calibrate (or reuse the persisted calibration) --------------------------------------
    const SoakSummary soak = runSoak(cfg.soakSeconds, mainBuffer);
    if (soak.aborted) {
        std::cout << "[Abort] Ctrl+C during soak.\n";
        stopWorkers();
        if (!g_highResTimerAvailable) timeEndPeriod(1);
        appendMetadata(cfg.metadataPath, "session_end              : " + isoLocalNow() + " (aborted during soak)\n");
        return 130;
    }
    if (cfg.soakSeconds == 0) std::cout << "[WARN] --soak-seconds 0: results are NOT steady-state.\n";

    std::string calSource;
    if (needCalibration) {
        if (!runCalibration(mainBuffer, cfg.soakSeconds, cal)) {
            std::cout << "[Abort] Ctrl+C during calibration.\n";
            stopWorkers();
            if (!g_highResTimerAvailable) timeEndPeriod(1);
            appendMetadata(cfg.metadataPath, "session_end              : " + isoLocalNow() + " (aborted during calibration)\n");
            return 130;
        }
        g_loopCount.store(cal.loopCount);
        if (!saveCalibration(cfg.calibrationPath, cal)) std::cerr << "[WARN] could not write " << cfg.calibrationPath << "\n";
        calSource = "new calibration this session";
    }
    else {
        calSource = "loaded from " + cfg.calibrationPath;
    }
    std::cout << "[Calibration] loop count " << cal.loopCount << " (" << calSource << ", calibrated at " << cal.calibratedAt
        << ", median " << cal.measuredMedianMs << " ms, LP_sib-parked reference " << cal.siblingParkedRefMs << " ms)\n\n";
    {
        std::ostringstream m;
        m << std::fixed << std::setprecision(3)
          << "soak_seconds             : " << soak.seconds << " (first " << SOAK_REPORT_WINDOW_S << " s median "
          << soak.firstWindowMedianMs << " ms -> last window median " << soak.lastWindowMedianMs << " ms at the pre-calibration loop count)\n"
          << "calibrated_loop_count    : " << cal.loopCount << " (" << calSource << ")\n"
          << "calibration_timestamp    : " << cal.calibratedAt << "\n"
          << "calibration_median_ms    : " << cal.measuredMedianMs << "\n"
          << "sibling_parked_ref_ms    : " << cal.siblingParkedRefMs << " (P_SMT and SMT_IDLE drift reference)\n"
          << "calibration_soak_seconds : " << cal.soakSeconds << "\n";
        appendMetadata(cfg.metadataPath, m.str());
    }

    // ---- 5. CSV -------------------------------------------------------------------------------------------
    std::ofstream csv;
    if (!resume.hasContent) {
        csv.open(cfg.csvPath, std::ios::binary | std::ios::trunc);
        if (csv.is_open()) csv << csvHeaderLine() << "\n";
    }
    else {
        csv.open(cfg.csvPath, std::ios::binary | std::ios::app);
        if (csv.is_open() && !resume.endsWithNewline) csv << "\n"; // isolate a partial last line
    }
    if (!csv.is_open()) {
        std::cerr << "[ERROR] cannot open " << cfg.csvPath << " for writing\n";
        stopWorkers();
        return 1;
    }
    csv << std::fixed << std::setprecision(4);
    csv.flush();

    // Trace files: appended across resumed sessions; a header is written only to a new or empty file. Rows of a
    // test whose results row was never written (a crash between the two) have no matching Trace_ID.
    std::ofstream framesTrace, batchesTrace;
    if (!cfg.tracePrefix.empty()) {
        auto openTrace = [](std::ofstream& f, const std::string& path, const char* header) {
            std::error_code ec;
            const bool fresh = !fs::exists(path, ec) || fs::file_size(path, ec) == 0;
            f.open(path, std::ios::binary | std::ios::app);
            if (f.is_open() && fresh) f << header << "\n";
            return f.is_open();
        };
        const std::string fp = cfg.tracePrefix + "_frames.csv", bp = cfg.tracePrefix + "_batches.csv";
        if (!openTrace(framesTrace, fp, "Trace_ID,Frame,Start_ns,End_ns") ||
            !openTrace(batchesTrace, bp, "Trace_ID,Batch,Release_ns,Start_ns,End_ns,Missed,LP_Start,LP_End,Thread_Cycles")) {
            std::cerr << "[ERROR] cannot open the trace files " << fp << " / " << bp << " for writing\n";
            stopWorkers();
            return 1;
        }
    }

    // ---- 6. Tests -----------------------------------------------------------------------------------------
    size_t written = 0, skipped = 0, driftFlags = 0, truncatedCount = 0, perAgentFallbacks = 0, processed = 0;
    double executedSeconds = 0.0;
    std::set<std::string> perAgentModels;

    for (const TestCase& tc : plan) {
        if (isDone(tc)) continue;
        if (g_abortRequested.load()) break;
        if (cfg.maxTests >= 0 && static_cast<int>(written) >= cfg.maxTests) {
            std::cout << "\n[Session] --max-tests " << cfg.maxTests << " reached.\n";
            break;
        }
        bool acNow = false;
        acStatusString(&acNow);
        if (!acNow && !cfg.allowBattery) {
            std::cerr << "\n[ERROR] AC power lost; stopping the session (re-run to resume).\n";
            exitCode = 4;
            break;
        }

        ++processed;
        std::cout << "\n[" << processed << "/" << remainingAtStart << "] In " << tc.inputs << " | Hid " << tc.hidden
            << " | Lay " << tc.layers << " | Agt " << tc.density << " | " << conditionName(tc.cond) << " | Trial " << tc.trial
            << " | block " << tc.block << " pos " << tc.blockPos << "\n";

        const auto testStart = Clock::now();
        TestRow row;
        std::string message, warning;
        const TestStatus status = runTest(tc, cfg, env, cal, mainBuffer, row, message, warning);

        if (!warning.empty()) {
            const std::string model = modelFileName(cfg, tc);
            std::cout << "  [WARN] " << model << ": " << warning << "\n";
            std::cerr << "[WARN] " << model << ": " << warning << "\n";
            if (perAgentModels.insert(model).second) ++perAgentFallbacks;
        }
        if (status == TestStatus::Aborted) {
            std::cout << "  [Abort] test discarded (nothing written).\n";
            break;
        }
        if (status == TestStatus::Skipped) {
            ++skipped;
            std::cerr << "[ERROR] " << conditionName(tc.cond) << " in" << tc.inputs << "_H" << tc.hidden << "_L" << tc.layers
                << " density " << tc.density << " trial " << tc.trial << ": " << message << " -- nothing written, continuing.\n";
            continue;
        }

        char traceId[48];
        snprintf(traceId, sizeof(traceId), "%s-%04zu", runId.c_str(), processed);
        if (!cfg.tracePrefix.empty()) writeTraces(framesTrace, batchesTrace, traceId, row);
        csv << buildCsvRow(runId, traceId, tc, row) << "\n";
        csv.flush();
        ++written;
        if (row.driftFlag) ++driftFlags;
        if (row.truncated) ++truncatedCount;

        const double testS = secondsSince(testStart);
        executedSeconds += testS;
        std::cout << "  Logged: " << row.game.n << " frames (median " << row.game.median << " ms, P99 " << row.game.p99 << " ms)";
        if (hasAIThread(tc.cond)) {
            std::cout << " | " << row.ai.n << " AI batches [" << row.batchMode << "] (median " << row.ai.median << " ms, missed "
                << row.missed << ", duty " << std::setprecision(3) << row.duty << std::setprecision(2) << ", start LPs "
                << row.startLPs << ")";
        }
        std::cout << " | window " << row.windowS << " s" << (row.truncated ? " TRUNCATED" : "") << "\n";
        const size_t left = remainingAtStart - processed;
        const double meanS = executedSeconds / static_cast<double>(written);
        std::cout << "  ETA: " << formatDuration(meanS * static_cast<double>(left)) << " remaining (" << left
            << " tests left, mean " << meanS << " s/test)\n";
    }

    if (g_abortRequested.load() && exitCode == 0) exitCode = 130;

    // ---- 7. Shutdown: join the persistent workers --------------------------------------------------------
    csv.close();
    stopWorkers();
    if (!g_highResTimerAvailable) timeEndPeriod(1);

    std::cout << "\n[Session] written " << written << ", skipped " << skipped << ", drift flags " << driftFlags
        << ", truncated " << truncatedCount << ", per-agent fallback models " << perAgentFallbacks << "\n";
    if (perAgentFallbacks > 0)
        std::cout << "[WARN] " << perAgentFallbacks << " model(s) ran in per_agent mode; see the Batch_Mode column.\n";
    {
        std::ostringstream m;
        m << "session_end              : " << isoLocalNow() << " exit_code=" << exitCode << " written=" << written
          << " skipped=" << skipped << " drift_flags=" << driftFlags << " truncated=" << truncatedCount
          << " per_agent_fallback_models=" << perAgentFallbacks << "\n";
        appendMetadata(cfg.metadataPath, m.str());
    }
    std::cout << (exitCode == 0 ? "\nSession complete.\n" : "\nSession stopped early.\n");
    return exitCode;
}
