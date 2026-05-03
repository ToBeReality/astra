#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/env_astra.sh" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/env_astra.sh"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/lmms-eval"
export DECORD_NUM_THREADS=8

accelerate launch \
  --main_process_port 18888 \
  --num_processes 8 \
  -m lmms_eval \
  --model llava_onevision \
  --model_args "pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=32,attn_implementation=flash_attention_2,video_decode_backend=decord,enable_astra=True,retention_ratio=0.05,expansion=1.25,token_selection_method=visual_guided_pruning,post_merge_selection_method=semantic_recycle_pruning,frame_selection_method=stv_guided_dynamic_budget_allocation,temporal_threshold=0.8,post_merge_triple_dom_parts=7,post_merge_triple_visual_cluster_parts=1,post_merge_triple_sem_parts=2,stv_budget_mu_low=0.05,stv_budget_min_tokens_per_frame=1,stv_budget_temperature=0.45,stv_budget_mu_threshold=0.10,stv_budget_mu_full=0.30,stv_budget_uniform_cv_threshold=0.08,pruning_layer=20,llm_retention_ratio=0.3" \
  --tasks "videomme,mvbench,longvideobench_val_v,mlvu_test" \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix astra_llava_onevision \
  --output_path "${SCRIPT_DIR}/logs"
