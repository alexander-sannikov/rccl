#include "nccl/net.h"

#include <dlfcn.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
#include <unistd.h>
#include <stdbool.h>
#include <valgrind/callgrind.h>


#define _STRINGIFY(x) #x
#define STRINGIFY(x)  _STRINGIFY(x)
#define EMIT_SSC_MARK( MARK_ID )                     \
        __asm__ __volatile__ (                       \
        "\n\t  movl $" STRINGIFY(MARK_ID) ", %%ebx"  \
        "\n\t  .byte 0x64, 0x67, 0x90"               \
        : : : "%ebx", "memory" )



#define HIST_BUCKETS    20
#define HIST_BASE_SHIFT 6   /* bucket 0: < 2^6 (< 64 ticks) */

typedef struct {
    uint64_t min_ns;
    uint64_t max_ns;
    uint64_t sum_ns;
    uint64_t count;
    uint64_t hist[HIST_BUCKETS];
} TimingStats;

typedef struct {
    void (*region_begin)(int idx);
    void (*region_end)(int idx);
} prof_iface;

#define NUM_REGIONS 8

static ncclNet_v10_t realPlugin;
static void* realPluginLib;

static TimingStats sendStats;
static TimingStats recvStats;
static TimingStats testNoCompStats;
static TimingStats testCompStats;
static TimingStats regionStats[NUM_REGIONS];
static uint64_t regionStart[NUM_REGIONS];

static void statsInit(TimingStats* s) {
    s->min_ns = UINT64_MAX;
    s->max_ns = 0;
    s->sum_ns = 0;
    s->count = 0;
    memset(s->hist, 0, sizeof(s->hist));
}

__attribute__((constructor))
static void profConstruct(void) {
    statsInit(&sendStats);
    statsInit(&recvStats);
    statsInit(&testNoCompStats);
    statsInit(&testCompStats);
    for (int i = 0; i < NUM_REGIONS; i++)
        statsInit(&regionStats[i]);
}

static void statsRecord(TimingStats* s, uint64_t ns) {
    if (ns < s->min_ns) s->min_ns = ns;
    if (ns > s->max_ns) s->max_ns = ns;
    s->sum_ns += ns;
    s->count++;
    int bucket = 0;
    if (ns >= (1ULL << HIST_BASE_SHIFT)) {
        bucket = 63 - __builtin_clzll(ns) - HIST_BASE_SHIFT + 1;
        if (bucket >= HIST_BUCKETS) bucket = HIST_BUCKETS - 1;
    }
    s->hist[bucket]++;
}

static inline uint64_t nowTicks(void) {
    return __rdtsc();
}

static void profRegionBegin(int idx) {
    if (idx >= 0 && idx < NUM_REGIONS)
        regionStart[idx] = nowTicks();
}

static void profRegionEnd(int idx) {
    if (idx >= 0 && idx < NUM_REGIONS)
        statsRecord(&regionStats[idx], nowTicks() - regionStart[idx]);
}

/* Timing profiling wrappers */
static ncclResult_t profIsend_timing(void* sendComm, void* data, size_t size,
                              int tag, void* mhandle, void* phandle,
                              void** request) {
    uint64_t t0 = nowTicks();
    ncclResult_t ret = realPlugin.isend(sendComm, data, size, tag,
                                        mhandle, phandle, request);
    statsRecord(&sendStats, nowTicks() - t0);
    return ret;
}

static ncclResult_t profIrecv_timing(void* recvComm, int n, void** data,
                              size_t* sizes, int* tags, void** mhandles,
                              void** phandles, void** request) {
    uint64_t t0 = nowTicks();
    ncclResult_t ret = realPlugin.irecv(recvComm, n, data, sizes, tags,
                                        mhandles, phandles, request);
    statsRecord(&recvStats, nowTicks() - t0);
    return ret;
}

static ncclResult_t profTest_timing(void* request, int* done, int* sizes) {
    uint64_t t0 = nowTicks();
    ncclResult_t ret = realPlugin.test(request, done, sizes);
    uint64_t elapsed = nowTicks() - t0;
    if (ret == ncclSuccess && done && *done)
        statsRecord(&testCompStats, elapsed);
    else
        statsRecord(&testNoCompStats, elapsed);
    return ret;
}

static const char* histBucketLabel(int bucket) {
    static const char* labels[HIST_BUCKETS] = {
        "<64",   "64-128", "128-256", "256-512",
        "512-1K","1-2K",   "2-4K",   "4-8K",
        "8-16K", "16-32K", "32-64K", "64-128K",
        "128-256K","256-512K","512K-1M","1-2M",
        "2-4M",  "4-8M",   "8-16M",  ">=16M"
    };
    return labels[bucket];
}

#define BUF_SIZE 4096

static int bprintf(char* buf, int off, int max, const char* fmt, ...)
    __attribute__((format(printf, 4, 5)));

static int bprintf(char* buf, int off, int max, const char* fmt, ...) {
    if (off >= max) return off;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + off, max - off, fmt, ap);
    va_end(ap);
    return off + (n > 0 ? n : 0);
}

static int printHist(char* buf, int off, int max, const TimingStats* s) {
    off = bprintf(buf, off, max, " {");
    int first = 1;
    for (int i = 0; i < HIST_BUCKETS; i++) {
        if (s->hist[i] == 0) continue;
        if (!first) off = bprintf(buf, off, max, ", ");
        off = bprintf(buf, off, max, "%s:%" PRIu64, histBucketLabel(i), s->hist[i]);
        first = 0;
    }
    return bprintf(buf, off, max, "}");
}

static int printStats(char* buf, int off, int max, const TimingStats* s, int withCount) {
    if (s->count > 0) {
        uint64_t avg = s->sum_ns / s->count;
        off = bprintf(buf, off, max, "<%" PRIu64 ", %" PRIu64 ", %" PRIu64,
                      s->min_ns, s->max_ns, avg);
    } else {
        off = bprintf(buf, off, max, "<0, 0, 0");
    }
    if (withCount)
        off = bprintf(buf, off, max, ", %" PRIu64 ">", s->count);
    else
        off = bprintf(buf, off, max, ">");
    return off;
}

static int statsPrinted = 0;

static void profPrintStats(void) {
    if (statsPrinted)
        return;
    statsPrinted = 1;

    char buf[BUF_SIZE];
    int off = 0;

    off = bprintf(buf, off, BUF_SIZE, "%d recv time: ", getpid());
    off = printStats(buf, off, BUF_SIZE, &recvStats, 0);
    if (recvStats.count) off = printHist(buf, off, BUF_SIZE, &recvStats);

    off = bprintf(buf, off, BUF_SIZE, ", send time: ");
    off = printStats(buf, off, BUF_SIZE, &sendStats, 0);
    if (sendStats.count) off = printHist(buf, off, BUF_SIZE, &sendStats);

    off = bprintf(buf, off, BUF_SIZE, ", test time (no completion): ");
    off = printStats(buf, off, BUF_SIZE, &testNoCompStats, 1);
    if (testNoCompStats.count) off = printHist(buf, off, BUF_SIZE, &testNoCompStats);

    off = bprintf(buf, off, BUF_SIZE, ", test time (with completion): ");
    off = printStats(buf, off, BUF_SIZE, &testCompStats, 1);
    if (testCompStats.count) off = printHist(buf, off, BUF_SIZE, &testCompStats);

    for (int i = 0; i < NUM_REGIONS; i++) {
        if (regionStats[i].count > 0) {
            off = bprintf(buf, off, BUF_SIZE, ", region[%d]: ", i);
            off = printStats(buf, off, BUF_SIZE, &regionStats[i], 1);
            off = printHist(buf, off, BUF_SIZE, &regionStats[i]);
        }
    }

    off = bprintf(buf, off, BUF_SIZE, "\n");
    write(STDERR_FILENO, buf, off < BUF_SIZE ? off : BUF_SIZE);
}

static ncclResult_t profCloseSend(void* sendComm) {
    profPrintStats();
    return realPlugin.closeSend(sendComm);
}

static ncclResult_t profCloseRecv(void* recvComm) {
    profPrintStats();
    return realPlugin.closeRecv(recvComm);
}

static ncclResult_t profCloseListen(void* listenComm) {
    profPrintStats();
    return realPlugin.closeListen(listenComm);
}


/* SDE profiling wrappers */
static ncclResult_t profIsend_sde(void* sendComm, void* data, size_t size,
    int tag, void* mhandle, void* phandle,
    void** request) {
    EMIT_SSC_MARK(1);
    ncclResult_t ret = realPlugin.isend(sendComm, data, size, tag,
                mhandle, phandle, request);
    EMIT_SSC_MARK(2);
    return ret;
}

static ncclResult_t profIrecv_sde(void* recvComm, int n, void** data,
    size_t* sizes, int* tags, void** mhandles,
    void** phandles, void** request) {
    EMIT_SSC_MARK(3);
    ncclResult_t ret = realPlugin.irecv(recvComm, n, data, sizes, tags,
              mhandles, phandles, request);
    EMIT_SSC_MARK(4);
    return ret;
}

static ncclResult_t profTest_sde(void* request, int* done, int* sizes) {
    EMIT_SSC_MARK(5);
    ncclResult_t ret = realPlugin.test(request, done, sizes);
    EMIT_SSC_MARK(6);
    return ret;
}

/*callgrind profiling wrappers*/
static ncclResult_t profIsend_callgrind(void* sendComm, void* data, size_t size,
    int tag, void* mhandle, void* phandle,
    void** request) {
    CALLGRIND_START_INSTRUMENTATION;
    ncclResult_t ret = realPlugin.isend(sendComm, data, size, tag,
                mhandle, phandle, request);
    CALLGRIND_STOP_INSTRUMENTATION;
    return ret;
}

static ncclResult_t profIrecv_callgrind(void* recvComm, int n, void** data,
    size_t* sizes, int* tags, void** mhandles,
    void** phandles, void** request) {
    CALLGRIND_START_INSTRUMENTATION;
    ncclResult_t ret = realPlugin.irecv(recvComm, n, data, sizes, tags,
                mhandles, phandles, request);
    CALLGRIND_STOP_INSTRUMENTATION;
    return ret;
}

static ncclResult_t profTest_callgrind(void* request, int* done, int* sizes) {
    CALLGRIND_START_INSTRUMENTATION;
    ncclResult_t ret = realPlugin.test(request, done, sizes);
    CALLGRIND_STOP_INSTRUMENTATION;
    return ret;
}

static ncclResult_t profInit(ncclDebugLogger_t logFunction,
                             ncclProfilerCallback_t profFunction);
static ncclResult_t profCloseSend(void* sendComm);
static ncclResult_t profCloseRecv(void* recvComm);
static ncclResult_t profCloseListen(void* listenComm);

ncclNet_v10_t ncclNetPlugin_v10 = {
    .name = "Profiler",
    .init = profInit,
};

int getMyMpiRank() {
    static const char* vars[] = {
        "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK",
        "PMI_RANK", "SLURM_PROCID", NULL
    };
    for (const char** v = vars; *v; v++) {
        const char* val = getenv(*v);
        if (val) return atoi(val);
    }
    return -1;
}

/* Environment variable: RCCL_NET_PROFILE_CONF=<type:{sde,timing,callgrind}>:<functions:{isend,irecv,test,all}>:<rank1,rank2,...>*/
bool enableProfile = false;
bool useSDE = false;
bool useTiming = false;
bool useCallgrind = false;
bool doProfIsend = false;
bool doProfIrecv = false;
bool doProfTest = false;

static void parseProfileConf() {
    const char* env = getenv("RCCL_NET_PROFILE_CONF");
    if (!env || !*env) {
        return;
    }
    char* conf = strdup(env);
    if (!conf) return;
    char* type = strtok(conf, ":");
    char* function = strtok(NULL, ":");
    char* ranks = strtok(NULL, ":");
    if (ranks) {
        char* rank = strtok(ranks, ",");
        while (rank) {
            int rankNum = atoi(rank);
            if (rankNum == getMyMpiRank()) {
                enableProfile = true;
            }
            rank = strtok(NULL, ",");
        }
    }

    if (!enableProfile) {
        return;
    }
    if (type) {
        if (strcmp(type, "sde") == 0) {
            useSDE = true;
        }
        else if (strcmp(type, "timing") == 0) {
            useTiming = true;
        }
        else if (strcmp(type, "callgrind") == 0) {
            if (RUNNING_ON_VALGRIND) {
                useCallgrind = true;
            } else {
                fprintf(stderr, "net_profiler: callgrind is only supported on Valgrind\n");
            }
        }
    }
    if (function) {
        char* tok = strtok(function, ",");
        while (tok) {
            if (strcmp(tok, "all") == 0) {
                doProfIsend = doProfIrecv = doProfTest = true;
            } else if (strcmp(tok, "isend") == 0) {
                doProfIsend = true;
            } else if (strcmp(tok, "irecv") == 0) {
                doProfIrecv = true;
            } else if (strcmp(tok, "test") == 0) {
                doProfTest = true;
            }
            tok = strtok(NULL, ",");
        }
    }
    free(conf);
}

const char* builtInNetPlugins[] = { "ncclNetIb", "rocmNetIb", NULL };
const char* builtInNetPluginNames[] = { "IB", "ROCM-IB", NULL };

const char* getBuiltInNetPluginName(const char* pluginName) {
    for (int i = 0; builtInNetPlugins[i]; i++) {
        if (strcmp(pluginName, builtInNetPlugins[i]) == 0) {
            return builtInNetPluginNames[i];
        }
    }
    return NULL;
}

static ncclResult_t profInit(ncclDebugLogger_t logFunction,
                             ncclProfilerCallback_t profFunction) {
    const char* path = getenv("RCCL_REAL_NET_PLUGIN");
    if (!path || !*path) {
        fprintf(stderr, "net_profiler: RCCL_REAL_NET_PLUGIN env var not set\n");
        return ncclInternalError;
    }
    ncclNet_v10_t* real = NULL;
    const char* builtInNetPluginName = getBuiltInNetPluginName(path);
    if (! builtInNetPluginName) {
        realPluginLib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!realPluginLib) {
            fprintf(stderr, "net_profiler: dlopen(%s) failed: %s\n",
                    path, dlerror());
            return ncclInternalError;
        }
        real = (ncclNet_v10_t*)dlsym(realPluginLib, "ncclNetPlugin_v10");
        if (!real) {
            fprintf(stderr, "net_profiler: ncclNetPlugin_v10 not found in %s\n",
                    path);
            dlclose(realPluginLib);
            realPluginLib = NULL;
            return ncclInternalError;
        }
    } else {
        real = (ncclNet_v10_t*)dlsym(RTLD_DEFAULT, builtInNetPluginName);
        if (!real) {
            fprintf(stderr, "net_profiler: %s not found\n", builtInNetPluginName);
            return ncclInternalError;
        }
    }

    memcpy(&realPlugin, real, sizeof(ncclNet_v10_t));
    memcpy(&ncclNetPlugin_v10, real, sizeof(ncclNet_v10_t));

    bool useTiming = false;
    bool useSDE = false;
    bool useCallgrind = false;

    if (useTiming) {
        if (doProfIsend) ncclNetPlugin_v10.isend = profIsend_timing;
        if (doProfIrecv) ncclNetPlugin_v10.irecv = profIrecv_timing;
        if (doProfTest) ncclNetPlugin_v10.test = profTest_timing;
        ncclNetPlugin_v10.closeSend = profCloseSend;
        ncclNetPlugin_v10.closeRecv = profCloseRecv;
    } else if (useSDE) {
        statsPrinted = 1;
        if (doProfIsend) ncclNetPlugin_v10.isend = profIsend_sde;
        if (doProfIrecv) ncclNetPlugin_v10.irecv = profIrecv_sde;
        if (doProfTest) ncclNetPlugin_v10.test = profTest_sde;
    } else if (useCallgrind) {
        statsPrinted = 1;       
        if (doProfIsend) ncclNetPlugin_v10.isend = profIsend_callgrind;
        if (doProfIrecv) ncclNetPlugin_v10.irecv = profIrecv_callgrind;
        if (doProfTest) ncclNetPlugin_v10.test = profTest_callgrind;
    }

    return realPlugin.init(logFunction, profFunction);
}

__attribute__((destructor))
static void profDestroy(void) {
    profPrintStats();

    if (realPluginLib) {
        dlclose(realPluginLib);
        realPluginLib = NULL;
    }
}
