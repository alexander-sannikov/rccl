# RCCL Net Profiler Plugin

An interposition plugin for RCCL's `ncclNet_v10` network interface that wraps a
real net plugin (external `.so` or built-in) and instruments `isend`, `irecv`,
and `test` calls with one of three profiling backends.

## Building

```
make            # produces libnccl-net-profiler.so
make clean      # removes the shared object
```

Requirements: a C compiler, `valgrind/callgrind.h` headers (for the callgrind
backend), and the NCCL net headers in `../example/nccl/`.

## Configuration

Two environment variables control the plugin:

### `RCCL_REAL_NET_PLUGIN`

Path to the real net plugin to wrap. This can be:

- An absolute path to a shared library (e.g. `/usr/lib/libnccl-net.so`), which
  is loaded with `dlopen`.
- A built-in plugin name (`ncclNetIb` or `rocmNetIb`), which is looked up with
  `dlsym(RTLD_DEFAULT, ...)` in the already-loaded process image.

### `RCCL_NET_PROFILE_CONF`

Format: `<type>:<function>:<ranks>`

| Field      | Values                              | Description                                           |
|------------|-------------------------------------|-------------------------------------------------------|
| `type`     | `timing`, `sde`, `callgrind`        | Profiling backend (see below)                         |
| `function` | `isend`, `irecv`, `test`, `all`     | Which net operations to instrument                    |
| `ranks`    | Comma-separated integers            | MPI ranks to profile (others pass through uninstrumented) |

MPI rank is auto-detected from `OMPI_COMM_WORLD_RANK`, `MV2_COMM_WORLD_RANK`,
`PMI_RANK`, or `SLURM_PROCID`.

**Example:**

```bash
export RCCL_REAL_NET_PLUGIN=/usr/lib/libnccl-net.so
export RCCL_NET_PROFILE_CONF=timing:all:0,1
```

This instruments all three operations on ranks 0 and 1 using the timing backend.

## Usage

Point RCCL at the profiler plugin instead of the real plugin:

```bash
export NCCL_NET_PLUGIN=libnccl-net-profiler.so
export RCCL_REAL_NET_PLUGIN=/path/to/real/plugin.so
export RCCL_NET_PROFILE_CONF=timing:irecv:0

mpirun -np 8 ./my_app
```

## Profiling Backends

### `timing` — TSC cycle-counter statistics

Wraps each instrumented call with `rdtsc` reads and records per-call cycle
counts. Statistics are printed to stderr when a communicator is closed or at
process exit:

```
<pid> recv time: <min, max, avg> {histogram}, send time: <min, max, avg> {histogram}, ...
```

For `test`, completions and non-completions are tracked separately.

**Histogram buckets** are log2-scaled starting at 64 ticks:

```
<64, 64-128, 128-256, ... 8-16M, >=16M
```

Up to 8 user-defined regions can also be tracked via the `profRegionBegin` /
`profRegionEnd` callbacks (indices 0--7).

### `sde` — Intel SDE SSC marks

Emits SSC marker pairs around each instrumented call. These marks are recognised
by Intel Software Development Emulator for region-of-interest analysis:

| Operation | Start mark | End mark |
|-----------|-----------|----------|
| `isend`   | 1         | 2        |
| `irecv`   | 3         | 4        |
| `test`    | 5         | 6        |

No stats are printed — use SDE's own analysis tools to process the marks.

### `callgrind` — Valgrind/Callgrind instrumentation control

Enables Callgrind instrumentation only for the duration of each profiled call
using `CALLGRIND_START_INSTRUMENTATION` / `CALLGRIND_STOP_INSTRUMENTATION`.
This backend is only activated when the process is running under Valgrind;
otherwise a warning is printed and no instrumentation occurs.

Typical usage:

```bash
mpirun -np 8  \
    -x NCCL_NET_PLUGIN=libnccl-net-profiler.so \
    -x RCCL_REAL_NET_PLUGIN=/path/to/plugin.so \
    -x RCCL_NET_PROFILE_CONF=callgrind:irecv:0 \
    wrapper.sh <application>
```

## Wrapper Script (`wrapper.sh`)

A per-rank launcher that constructs `RCCL_NET_PROFILE_CONF` from command-line
options and, for profiled ranks, runs the application under the appropriate
external tool. Non-profiled ranks execute the application directly.

### Options

| Option    | Default  | Description                                              |
|-----------|----------|----------------------------------------------------------|
| `--mode`  | `timing` | Profiling backend: `timing`, `sde`, or `callgrind`       |
| `--func`  | `all`    | Comma-separated list: `isend`, `irecv`, `test`, or `all` |
| `--ranks` | `0`      | Comma-separated MPI ranks to profile                     |

Options are followed by `--` and then the application command line.

### Behavior by mode

- **`timing`** — exports `RCCL_NET_PROFILE_CONF` and runs the application
  directly. The profiler plugin handles `rdtsc` measurement internally.
- **`callgrind`** — profiled ranks are launched under
  `valgrind --tool=callgrind` with `--collect-atstart=no`, `--cache-sim=yes`,
  `--collect-jumps=yes`, and `--dump-instr=yes`. The plugin's
  `CALLGRIND_START/STOP_INSTRUMENTATION` macros control when data is collected.
- **`sde`** — profiled ranks are launched under Intel SDE (`sde64`) with
  `-mix` and the SSC start/stop mark pairs that match the selected functions.

### Environment variables

| Variable          | Default  | Description                              |
|-------------------|----------|------------------------------------------|
| `SDE`             | `sde64`  | Path to the Intel SDE binary             |
| `CALLGRIND_OUT_DIR` | `.`    | Directory for callgrind output files      |

### Examples

Profile `irecv` with timing on rank 0 (default settings):

```bash
mpirun -np 8 \
    -x NCCL_NET_PLUGIN=libnccl-net-profiler.so \
    -x RCCL_REAL_NET_PLUGIN=/path/to/plugin.so \
    ./wrapper.sh -- ./my_app
```

Profile `isend` and `irecv` under callgrind on ranks 0 and 1:

```bash
mpirun -np 8 \
    -x NCCL_NET_PLUGIN=libnccl-net-profiler.so \
    -x RCCL_REAL_NET_PLUGIN=/path/to/plugin.so \
    ./wrapper.sh --mode callgrind --func isend,irecv --ranks 0,1 -- ./my_app
```

Profile all functions under Intel SDE on rank 0:

```bash
mpirun -np 8 \
    -x NCCL_NET_PLUGIN=libnccl-net-profiler.so \
    -x RCCL_REAL_NET_PLUGIN=/path/to/plugin.so \
    ./wrapper.sh --mode sde --ranks 0 -- ./my_app
```

