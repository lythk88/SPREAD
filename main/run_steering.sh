# # PARAMS
MODEL_REPO="Qwen/Qwen2.5-Math-1.5B-Instruct"
MODEL_NAME="Qwen2.5-Math-1.5B-Instruct"
# MODEL_REPO="Qwen/Qwen2.5-1.5B"
# MODEL_NAME="Qwen2.5-1.5B"
OUTPUT_DIR=results/$(basename "$MODEL_REPO")$
START=0
END=1000
STEERN=500

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/math
mkdir -p $OUTPUT_DIR/gsm8k/
mkdir -p $OUTPUT_DIR/aime24
mkdir -p $OUTPUT_DIR/olympiadbench

# Run once with a large N for AIME24
TEMPERATURE=0.6
N=8
for LAYER in 21; do
  for CALPHA_K in 1; do
  echo "Running Steering algorithm1 on AIME24 dataset..."
  python main/run_steering.py \
    --model_repo "$MODEL_REPO" \
    --data_path "SPREAD/data/aime24/test.jsonl" \
    --input_start $START \
    --input_end $END \
    --number_candidate $N \
    --recalc_steer_after_n_tokens $STEERN \
    --temperature $TEMPERATURE \
    --calpha_k $CALPHA_K \
    --save_all_candidates \
    --steer_at_layer $LAYER \
    --output_dir $OUTPUT_DIR/aime24/steering \
    --run_name_after "steer${N}_aime24_steern${STEERN}-calpha-k${CALPHA_K}-temp${TEMPERATURE}-layer${LAYER}"
  done
done
python main/evaluate_strategies.py   --input main/results/$MODEL_NAME$/aime24/steering  --plot --dataset "aime24"   --output_dir main/results/$MODEL_NAME$/aime24/steering

# Run once with a large N for MATH500
N=4
for TEMPERATURE in 0.6; do
  for CALPHA_K in 1; do
  echo "Running Steering algorithm1 on MATH500 dataset..."
  python main/run_steering.py \
    --model_repo "$MODEL_REPO" \
    --data_path "SPREAD/data/math/test.jsonl" \
    --input_start $START \
    --input_end $END \
    --number_candidate $N \
    --recalc_steer_after_n_tokens $STEERN \
    --temperature $TEMPERATURE \
    --calpha_k $CALPHA_K \
    --save_all_candidates \
    --output_dir $OUTPUT_DIR/math500/steering \
    --run_name_after "steer${N}_math500_steern${STEERN}-calpha-k${CALPHA_K}-temp${TEMPERATURE}"
  done
  python main/evaluate_strategies.py   --input main/results/$MODEL_NAME$/math500/steering   --plot --dataset "math500"   --output_dir main/results/$MODEL_NAME$/math500/steering
done

# Run once with a large N for OLYMPIADBENCH
N=8
for TEMPERATURE in 0.4; do
  for CALPHA_K in 10; do
  echo "Running Steering algorithm1 on OlympiadBench dataset..."
  python main/run_steering.py \
    --model_repo "$MODEL_REPO" \
    --data_path "SPREAD/data/olympiadbench/test.jsonl" \
    --input_start $START \
    --input_end $END \
    --number_candidate $N \
    --recalc_steer_after_n_tokens $STEERN \
    --temperature $TEMPERATURE \
    --calpha_k $CALPHA_K \
    --save_all_candidates \
    --output_dir $OUTPUT_DIR/olympiadbench/steering \
    --run_name_after "steer${N}_olympiadbench_steern${STEERN}-calpha-k${CALPHA_K}-temp${TEMPERATURE}"
    python main/evaluate_strategies.py   --input main/results/$MODEL_NAME$/olympiadbench/steering   --plot --dataset "olympiadbench"   --output_dir main/results/$MODEL_NAME$/olympiadbench/steering
  done
done



