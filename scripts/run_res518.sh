#!/usr/bin/env bash
# Overnight 5-way comparison at 518x518 (the resolution that recovers PyTorch accuracy).
#   1 PyTorch fp16        (already have: eval_results_torch_518x518.json = 0.8126)
#   2 Hybrid TRT fp16     torch DINO + fp16 aggregator chain + torch camera head
#   3 Full  TRT fp16      DINO + fp16 chain + camera head, all TRT
#   4 Hybrid TRT INT8     torch DINO + INT8 aggregator chain + torch camera head
#   5 Full  TRT INT8      DINO + INT8 chain + camera head, all TRT
#
# Crash-safe: CPU/fp32 exports (cushioned by swap, cannot GPU-OOM), capped trtexec builds,
# expandable_segments:False evals. IDEMPOTENT: every step skips if its artifact/result already
# exists, so a crash + rerun resumes. Continues past per-config failures (full-TRT @518 evals
# are the memory-risky ones on the 7.6 GB board → ordered LAST so 1,2,4 finish first).
set -u
cd /workspace
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
TRTEXEC=/usr/src/tensorrt/bin/trtexec
ART=deploy/artifacts
H=518; W=518; F=10
CATS="apple"; MAXS=2
log(){ echo "=== [$(date +%H:%M:%S)] $* ==="; }

# export+fp16+build one engine (fp16 stronglyTyped; Q/DQ in the onnx ⇒ INT8 on those layers)
build_engine(){  # $1=onnx_path  $2=engine_name
  local onnx="$1" name="$2" eng="$ART/engines/$2.trt"
  [ -f "$eng" ] && { log "engine $name exists -> skip"; return 0; }
  [ -f "$onnx" ] || { log "MISSING onnx $onnx -> cannot build $name"; return 1; }
  local fp16="$ART/onnx/${name}_fp16/${name}_fp16.onnx"
  [ -f "$fp16" ] || python scripts/onnx_to_fp16.py --in "$onnx" --out "$fp16" || return 1
  sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
  log "trtexec build $name"
  "$TRTEXEC" --onnx="$fp16" --saveEngine="$eng" --stronglyTyped --maxAuxStreams=0 \
    --memPoolSize=workspace:512 --builderOptimizationLevel=3 --skipInference \
    --timingCacheFile="$ART/timing_cache/orin_sm87.cache" >"$ART/profiles/${name}_build.log" 2>&1 \
    || { log "BUILD FAILED $name (see profiles/${name}_build.log)"; return 1; }
}

# ============ Config 4: Hybrid INT8 @518 (uses the standalone CPU export's slices) ============
log "Wait for INT8@518 slices (standalone export) ..."
while [ ! -f "$ART/onnx/chain_modelopt_max_pc_518x518/chain_modelopt_max_pc_518x518_s16.onnx" ]; do sleep 60; done
while pgrep -f "export_modelopt_ptq.py.*518x518" >/dev/null 2>&1; do sleep 60; done
log "Config 4: Hybrid INT8 @518 (build INT8 chain engines + eval)"
python scripts/bench_trt_modelopt_chain.py --algorithm max --height $H --width $W \
  --categories $CATS --max_seqs $MAXS || log "config4 (hybrid INT8) FAILED"

# ============ fp16 aggregator chain @518 (CPU export) -> used by config 2 & 3 ============
for s in 0 8 16; do
  islast=""; [ "$s" = "16" ] && islast="--slice_is_last"
  onnx="$ART/onnx/chain_f10_518_s$s/chain_f10_518_s$s.onnx"
  if [ ! -f "$onnx" ]; then
    log "Export fp16 aggregator slice s$s @518 (CPU)"
    python scripts/trt_export_pose.py --precision slice --frames $F --slice_start $s \
      --slice_pairs 8 $islast --height $H --width $W --out "$onnx" || log "slice s$s export FAILED"
  fi
  build_engine "$onnx" "chain_f10_518_s$s" || true
done

# ============ Config 2: Hybrid fp16 @518 ============
if [ ! -f "$ART/results/eval_results_trt_chain_fp16_518.json" ]; then
  log "Config 2: Hybrid fp16 @518 (eval_chain)"
  python - <<PY || log "config2 (hybrid fp16) FAILED"
import sys; sys.path.insert(0,"scripts"); sys.path.insert(0,".")
import _trt_bench_common as C
engs=[C.engine_path(f"chain_f10_518_s{s}") for s in (0,8,16)]
C.eval_chain(engs, C.results_path("trt_chain_fp16_518"), label="Hybrid TRT fp16 @518",
             categories="$CATS", max_seqs=$MAXS, height=$H, width=$W, num_frames=$F)
PY
fi

# ============ DINO @518 (nofuse) + camera_head @518 (CPU export) -> used by config 3 & 5 ======
dbase="$ART/onnx/dino_split_518/dino_split_518"
if [ ! -f "${dbase}_1.onnx" ]; then
  log "Export DINO @518 (--no_fused_attn, CPU)"
  python scripts/trt_export_pose.py --precision dino_split --dino_chunks 2 --frames $F \
    --height $H --width $W --no_fused_attn --out "${dbase}.onnx" || log "DINO export FAILED"
fi
build_engine "${dbase}_0.onnx" "dino_0_nofuse_518" || true
build_engine "${dbase}_1.onnx" "dino_1_nofuse_518" || true

cbase="$ART/onnx/camera_head_518/camera_head_518.onnx"
if [ ! -f "$cbase" ]; then
  log "Export camera_head @518 (CPU)"
  python scripts/trt_export_pose.py --precision camera_head --frames $F --height $H --width $W \
    --out "$cbase" || log "camera_head export FAILED"
fi
build_engine "$cbase" "camera_head_518" || true

# ============ Config 5: Full all-TRT INT8 @518 (memory-risky) ============
log "Config 5: Full all-TRT INT8 @518"
python scripts/bench_trt_fulltrt.py --height $H --width $W --categories $CATS --max_seqs $MAXS \
  --label "Full TRT INT8 @518" --results "$ART/results/eval_results_trt_fulltrt_int8_518.json" \
  --engines dino_0_nofuse_518 dino_1_nofuse_518 \
            chain_modelopt_max_pc_518x518_s0 chain_modelopt_max_pc_518x518_s8 chain_modelopt_max_pc_518x518_s16 \
            camera_head_518 || log "config5 (full INT8) FAILED (likely memory @518)"

# ============ Config 3: Full all-TRT fp16 @518 (memory-risky) ============
log "Config 3: Full all-TRT fp16 @518"
python scripts/bench_trt_fulltrt.py --height $H --width $W --categories $CATS --max_seqs $MAXS \
  --label "Full TRT fp16 @518" --results "$ART/results/eval_results_trt_fulltrt_fp16_518.json" \
  --engines dino_0_nofuse_518 dino_1_nofuse_518 \
            chain_f10_518_s0 chain_f10_518_s8 chain_f10_518_s16 camera_head_518 \
  || log "config3 (full fp16) FAILED (likely memory @518)"

# ============ Comparison table ============
log "Aggregating comparison table"
python scripts/compare_results.py || true
log "ALL DONE — see deploy/artifacts/results/comparison.md"
