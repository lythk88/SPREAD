# # PARAMS
MODEL_REPO="Qwen/Qwen2.5-Math-1.5B-Instruct"
MODEL_NAME="Qwen2.5-Math-1.5B-Instruct"
# MODEL_REPO="Qwen/Qwen2.5-1.5B"
# MODEL_NAME="Qwen2.5-1.5B"
OUTPUT_DIR=main/results/$(basename "$MODEL_REPO")$
START=0
END=1000

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/math
mkdir -p $OUTPUT_DIR/aime24
mkdir -p $OUTPUT_DIR/olympiadbench

# Run once with a large N for AIME24
N=8
for TEMPERATURE in 0.2 0.4 0.6 0.8 1.0; do
  echo "Running BASELINE on AIME24 dataset..."
  python main/run_baseline.py\
    --model_repo "$MODEL_REPO" \
    --data_path "SPREAD/data/aime24/test.jsonl" \
    --input_start $START \
    --input_end $END \
    --number_candidate $N \
    --temperature $TEMPERATURE \
    --save_all_candidates \
    --output_dir $OUTPUT_DIR/aime24/baseline \
    --run_name_before "bestof8_aime24_temp${TEMPERATURE}"
  python main/evaluate_strategies.py   --input main/results/$MODEL_NAME$/aime24/baseline   --plot --dataset "aime24"   --output_dir main/results/$MODEL_NAME$/aime24/baseline
done

# Run once with a large N for MATH500
N=4
for TEMPERATURE in 0.2 0.4 0.6 0.8 1.0; do
  echo "Running BASELINE on MATH500 dataset..."
  python main/run_baseline.py\
    --model_repo "$MODEL_REPO" \
    --data_path "SPREAD/data/math/test.jsonl" \
    --input_start $START \
    --input_end $END \
    --number_candidate $N \
    --temperature $TEMPERATURE \
    --save_all_candidates \
    --output_dir $OUTPUT_DIR/math500/baseline \
    --run_name_before "bestof4_math500_temp${TEMPERATURE}"
  python main/evaluate_strategies.py   --input main/results/$MODEL_NAME$/math500/baseline   --plot --dataset "math500"   --output_dir main/results/$MODEL_NAME$/math500/baseline
done
# Run once with a large N for OLYMPIADBENCH
N=8
for TEMPERATURE in 0.2 0.4 0.6 0.8 1.0; do
  echo "Running BASELINE on OLYMPIAD dataset..."
  python main/run_baseline.py\
    --model_repo "$MODEL_REPO" \
    --data_path "SPREAD/data/olympiadbench/test.jsonl" \
    --input_start $START \
    --input_end $END \
    --number_candidate $N \
    --temperature $TEMPERATURE \
    --save_all_candidates \
    --output_dir $OUTPUT_DIR/olympiadbench/baseline \
    --run_name_before "bestof8_olympiadbench_temp${TEMPERATURE}"
  python main/evaluate_strategies.py   --input main/results/$MODEL_NAME$/olympiadbench/baseline   --plot --dataset "olympiadbench"   --output_dir main/results/$MODEL_NAME$/olympiadbench/baseline
done