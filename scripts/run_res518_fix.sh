#!/usr/bin/env bash
# Robust second pass for the 518x518 comparison. The first pass (run_res518.sh) produces all
# the ONNX (CPU exports) but the `is_last` slice engines OOM during the trtexec Myelin build
# at 518 when other work shares the GPU. This waits for the first pass to finish, then rebuilds
# every missing engine with a FREE GPU + low builder-opt-level (memory-frugal; same accuracy,
# only inference tactics differ), then re-runs the evals that lack a result. Idempotent.
set -u
cd /workspace
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
TRTEXEC=/usr/src/tensorrt/bin/trtexec
ART=deploy/artifacts
H=518; W=518; F=10; CATS="apple"; MAXS=2
log(){ echo "=== [$(date +%H:%M:%S)] FIX: $* ==="; }

log "Waiting for first pass (run_res518.sh) to finish ..."
while pgrep -f run_res518.sh >/dev/null 2>&1; do sleep 120; done
log "First pass done. Rebuilding missing engines on a free GPU (opt level 0)."

robust_build(){  # $1=onnx  $2=engine_name
  local onnx="$1" name="$2" eng="$ART/engines/$2.trt"
  [ -f "$eng" ] && { log "engine $name exists -> skip"; return 0; }
  [ -f "$onnx" ] || { log "MISSING onnx for $name: $onnx"; return 1; }
  local fp16="$ART/onnx/${name}_fp16/${name}_fp16.onnx"
  [ -f "$fp16" ] || python scripts/onnx_to_fp16.py --in "$onnx" --out "$fp16" || return 1
  sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
  log "robust trtexec build $name (opt0, ws1024)"
  "$TRTEXEC" --onnx="$fp16" --saveEngine="$eng" --stronglyTyped --maxAuxStreams=0 \
    --memPoolSize=workspace:1024 --builderOptimizationLevel=0 --skipInference \
    --timingCacheFile="$ART/timing_cache/orin_sm87.cache" >"$ART/profiles/${name}_fix.log" 2>&1 \
    || { log "STILL FAILED $name (see profiles/${name}_fix.log)"; return 1; }
}

MO=$ART/onnx/chain_modelopt_max_pc_518x518/chain_modelopt_max_pc_518x518
FP=$ART/onnx
# INT8 chain (config 4 + 5)
for s in 0 8 16; do robust_build "${MO}_s$s.onnx" "chain_modelopt_max_pc_518x518_s$s"; done
# fp16 chain (config 2 + 3)
for s in 0 8 16; do robust_build "$FP/chain_f10_518_s$s/chain_f10_518_s$s.onnx" "chain_f10_518_s$s"; done
# DINO + camera head (config 3 + 5)
robust_build "$FP/dino_split_518/dino_split_518_0.onnx" "dino_0_nofuse_518"
robust_build "$FP/dino_split_518/dino_split_518_1.onnx" "dino_1_nofuse_518"
robust_build "$FP/camera_head_518/camera_head_518.onnx" "camera_head_518"

# ---- re-run the evals whose result is missing (benches/eval skip already-built engines) ----
res(){ [ -f "$ART/results/$1" ]; }

if ! res eval_results_trt_modelopt_pc_max_518x518.json; then
  log "Config 4: Hybrid INT8 @518 (re-eval)"
  python scripts/bench_trt_modelopt_chain.py --algorithm max --height $H --width $W \
    --categories $CATS --max_seqs $MAXS || log "config4 re-eval FAILED"
fi

if ! res eval_results_trt_chain_fp16_518.json; then
  log "Config 2: Hybrid fp16 @518 (re-eval)"
  python - <<PY || log "config2 re-eval FAILED"
import sys; sys.path.insert(0,"scripts"); sys.path.insert(0,".")
import _trt_bench_common as C
engs=[C.engine_path(f"chain_f10_518_s{s}") for s in (0,8,16)]
C.eval_chain(engs, C.results_path("trt_chain_fp16_518"), label="Hybrid TRT fp16 @518",
             categories="$CATS", max_seqs=$MAXS, height=$H, width=$W, num_frames=$F)
PY
fi

if ! res eval_results_trt_fulltrt_int8_518.json; then
  log "Config 5: Full INT8 @518 (re-eval, memory-risky)"
  python scripts/bench_trt_fulltrt.py --height $H --width $W --categories $CATS --max_seqs $MAXS \
    --label "Full TRT INT8 @518" --results "$ART/results/eval_results_trt_fulltrt_int8_518.json" \
    --engines dino_0_nofuse_518 dino_1_nofuse_518 \
      chain_modelopt_max_pc_518x518_s0 chain_modelopt_max_pc_518x518_s8 chain_modelopt_max_pc_518x518_s16 \
      camera_head_518 || log "config5 re-eval FAILED"
fi

if ! res eval_results_trt_fulltrt_fp16_518.json; then
  log "Config 3: Full fp16 @518 (re-eval, memory-risky)"
  python scripts/bench_trt_fulltrt.py --height $H --width $W --categories $CATS --max_seqs $MAXS \
    --label "Full TRT fp16 @518" --results "$ART/results/eval_results_trt_fulltrt_fp16_518.json" \
    --engines dino_0_nofuse_518 dino_1_nofuse_518 \
      chain_f10_518_s0 chain_f10_518_s8 chain_f10_518_s16 camera_head_518 || log "config3 re-eval FAILED"
fi

log "Aggregating final comparison"
python scripts/compare_results.py || true
log "FIX PASS DONE."
