#!/bin/bash

# Test 3 LM parameter configurations
# Each run takes ~45 min on CPU

echo "Testing LM parameter configurations..."
echo "======================================="

# Config 1: Lower acoustic scale (trust LM more)
echo ""
echo "Config 1: acoustic=0.30, blank=85, alpha=0.60"
echo "Starting LM..."
python language_model/language-model-standalone.py \
    --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil \
    --nbest 100 \
    --acoustic_scale 0.30 \
    --blank_penalty 85 \
    --alpha 0.60 \
    --redis_ip localhost \
    --gpu_number -1 > /dev/null 2>&1 &

LM_PID=$!
sleep 15

echo "Running evaluation..."
python model_training/evaluate_model.py \
    --model_path data/t15_pretrained_rnn_baseline \
    --data_dir data/hdf5_data_final \
    --eval_type val \
    --gpu_number -1 \
    --csv_path data/t15_copyTaskData_description.csv \
    --use_session_norm 2>&1 | grep "Aggregate Word Error Rate"

kill $LM_PID
sleep 10

# Config 2: Higher acoustic scale (trust neural net more)  
echo ""
echo "Config 2: acoustic=0.35, blank=95, alpha=0.50"
echo "Starting LM..."
python language_model/language-model-standalone.py \
    --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil \
    --nbest 100 \
    --acoustic_scale 0.35 \
    --blank_penalty 95 \
    --alpha 0.50 \
    --redis_ip localhost \
    --gpu_number -1 > /dev/null 2>&1 &

LM_PID=$!
sleep 15

echo "Running evaluation..."
python model_training/evaluate_model.py \
    --model_path data/t15_pretrained_rnn_baseline \
    --data_dir data/hdf5_data_final \
    --eval_type val \
    --gpu_number -1 \
    --csv_path data/t15_copyTaskData_description.csv \
    --use_session_norm 2>&1 | grep "Aggregate Word Error Rate"

kill $LM_PID
sleep 10

# Config 3: Balanced
echo ""
echo "Config 3: acoustic=0.32, blank=88, alpha=0.55"
echo "Starting LM..."
python language_model/language-model-standalone.py \
    --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil \
    --nbest 100 \
    --acoustic_scale 0.32 \
    --blank_penalty 88 \
    --alpha 0.55 \
    --redis_ip localhost \
    --gpu_number -1 > /dev/null 2>&1 &

LM_PID=$!
sleep 15

echo "Running evaluation..."
python model_training/evaluate_model.py \
    --model_path data/t15_pretrained_rnn_baseline \
    --data_dir data/hdf5_data_final \
    --eval_type val \
    --gpu_number -1 \
    --csv_path data/t15_copyTaskData_description.csv \
    --use_session_norm 2>&1 | grep "Aggregate Word Error Rate"

kill $LM_PID

echo ""
echo "======================================="
echo "Testing complete!"