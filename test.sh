

set -e

set -o pipefail


GENERATE_SCRIPT="create_test_json.py"
MODEL_NAME="video"
TEMPERATURE=1
RELATIVE_POS=true
IS_EVAL=true
MODEL_PATH="model/model.cleanrl_model"
DIFF_PATH="data/max_movement.npy"



JSON_ROOT_PREDICTION="result_json/$MODEL_NAME"
IS_GT_PREDICTION=false


COMPOSITE_SCRIPT="composite_video.py"
COMPOSITE_SCRIPT_SINGLE="composite_video_single.py"
GT_RENDER_ROOT="render_gt" 
VIDEO_OUTPUT_ROOT="result_video"
FPS=4


TARGET_SAMPLES=(
"C01002747572"
)

for SAMPLE_NAME in "${TARGET_SAMPLES[@]}"; do
 
  
    python "$GENERATE_SCRIPT" \
        --target_sample_name "$SAMPLE_NAME" \
        --model_name "$MODEL_NAME" \
        --temperature "$TEMPERATURE" \
        --model_path "$MODEL_PATH" \
        --diff_path "$DIFF_PATH" \
        --relative_pos "$RELATIVE_POS" \
        --is_eval "$IS_EVAL"
    

    
    python vis/myvis_front.py --target_sample_name "$SAMPLE_NAME" --model_name "$MODEL_NAME" --json_root "$JSON_ROOT_PREDICTION" --is_gt "$IS_GT_PREDICTION"
    python vis/myvis_up.py --target_sample_name "$SAMPLE_NAME" --model_name "$MODEL_NAME" --json_root "$JSON_ROOT_PREDICTION" --is_gt "$IS_GT_PREDICTION"
    python vis/myvis_down.py --target_sample_name "$SAMPLE_NAME" --model_name "$MODEL_NAME" --json_root "$JSON_ROOT_PREDICTION" --is_gt "$IS_GT_PREDICTION"
    python vis/myvis_left.py --target_sample_name "$SAMPLE_NAME" --model_name "$MODEL_NAME" --json_root "$JSON_ROOT_PREDICTION" --is_gt "$IS_GT_PREDICTION"
    python vis/myvis_right.py --target_sample_name "$SAMPLE_NAME" --model_name "$MODEL_NAME" --json_root "$JSON_ROOT_PREDICTION" --is_gt "$IS_GT_PREDICTION"



    PREDICTION_RENDER_DIR="result_render/$SAMPLE_NAME/$MODEL_NAME"
    GT_RENDER_DIR="$GT_RENDER_ROOT/$SAMPLE_NAME/gt"
    VIDEO_OUTPUT_DIR="$VIDEO_OUTPUT_ROOT/$MODEL_NAME"
    

    mkdir -p "$VIDEO_OUTPUT_DIR"
    
    OUTPUT_VIDEO_PATH="$VIDEO_OUTPUT_DIR/${SAMPLE_NAME}.mp4"

    # python "$COMPOSITE_SCRIPT" \
    #     --prediction_dir "$PREDICTION_RENDER_DIR" \
    #     --gt_dir "$GT_RENDER_DIR" \
    #     --output_video_path "$OUTPUT_VIDEO_PATH" \
    #     --fps "$FPS"


done
echo ""
