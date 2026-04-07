#!/usr/bin/env bash
#
# Per-rank profiling wrapper for RCCL net plugins.
#
# Constructs RCCL_NET_PROFILE_CONF from command-line options, detects the
# current MPI rank, and — for profiled ranks — launches the application
# under the appropriate tool (valgrind/callgrind, Intel SDE, or plain
# execution for timing mode).
#
# Usage:
#   mpirun -np 8 ./wrapper.sh [options] -- <app> [app args ...]
#
# Options:
#   --mode <timing|sde|callgrind>   Profiling backend (default: timing)
#   --func <func[,func,...]>        Functions: isend,irecv,test,all (default: all)
#   --ranks <rank[,rank,...]>       MPI ranks to profile (default: 0)
#
# Optional environment:
#   SDE                 Path to Intel SDE binary (default: sde64)
#   CALLGRIND_OUT_DIR   Directory for callgrind output files (default: .)
#
# Examples:
#   mpirun -np 8 ./wrapper.sh -- ./my_app
#   mpirun -np 8 ./wrapper.sh --mode callgrind --func irecv --ranks 0,1 -- ./my_app
#   mpirun -np 8 ./wrapper.sh --mode sde --func isend,irecv --ranks 0 -- ./my_app
#

set -euo pipefail

die() { echo "wrapper.sh: error: $*" >&2; exit 1; }

mode=timing
func=all
ranks=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)  [[ $# -ge 2 ]] || die "--mode requires a value"; mode="$2"; shift 2 ;;
        --func)  [[ $# -ge 2 ]] || die "--func requires a value"; func="$2"; shift 2 ;;
        --ranks) [[ $# -ge 2 ]] || die "--ranks requires a value"; ranks="$2"; shift 2 ;;
        --)      shift; break ;;
        -*)      die "unknown option '$1'" ;;
        *)       break ;;
    esac
done

[[ $# -gt 0 ]] || die "no application specified"

case "$mode" in
    timing|sde|callgrind) ;;
    *) die "unknown mode '$mode' (expected timing|sde|callgrind)" ;;
esac

IFS=',' read -ra func_list <<< "$func"
for f in "${func_list[@]}"; do
    case "$f" in
        isend|irecv|test|all) ;;
        *) die "unknown function '$f' (expected isend|irecv|test|all)" ;;
    esac
done

export RCCL_NET_PROFILE_CONF="${mode}:${func}:${ranks}"

# ---------------------------------------------------------------------------
# Detect MPI rank from common environment variables.
# ---------------------------------------------------------------------------
get_my_rank() {
    for var in OMPI_COMM_WORLD_RANK MV2_COMM_WORLD_RANK PMI_RANK SLURM_PROCID; do
        val="${!var:-}"
        if [[ -n "$val" ]]; then
            echo "$val"
            return
        fi
    done
    echo "-1"
}

my_rank=$(get_my_rank)

rank_match=false
IFS=',' read -ra rank_list <<< "$ranks"
for r in "${rank_list[@]}"; do
    if [[ "$r" == "$my_rank" ]]; then
        rank_match=true
        break
    fi
done

if ! $rank_match; then
    exec "$@"
fi

# ---------------------------------------------------------------------------
# Rank is in the profiled set — launch under the appropriate tool.
# ---------------------------------------------------------------------------
echo "wrapper.sh [rank ${my_rank}]: mode=${mode} func=${func}" >&2

case "$mode" in
    timing)
        exec "$@"
        ;;

    callgrind)
        out_dir="${CALLGRIND_OUT_DIR:-.}"
        mkdir -p "$out_dir"

        exec valgrind \
            --tool=callgrind \
            --collect-atstart=no \
            --cache-sim=yes \
            --collect-jumps=yes \
            --dump-instr=yes \
            --callgrind-out-file="${out_dir}/callgrind.out.%p" \
            -- "$@"
        ;;

    sde)
        sde_marks=()
        for f in "${func_list[@]}"; do
            case "$f" in
                isend) sde_marks+=(-start_ssc_mark 1 -stop_ssc_mark 2) ;;
                irecv) sde_marks+=(-start_ssc_mark 3 -stop_ssc_mark 4) ;;
                test)  sde_marks+=(-start_ssc_mark 5 -stop_ssc_mark 6) ;;
                all)   sde_marks+=(-start_ssc_mark 1 -stop_ssc_mark 2
                                   -start_ssc_mark 3 -stop_ssc_mark 4
                                   -start_ssc_mark 5 -stop_ssc_mark 6) ;;
            esac
        done

        SDE="${SDE:-sde64}"
        exec "$SDE" -mix "${sde_marks[@]}" -- "$@"
        ;;
esac
