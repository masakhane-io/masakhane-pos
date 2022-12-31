for j in 1 2 3 4 5
do
  export MAX_LENGTH=200
  export BERT_MODEL=Davlan/afro-xlmr-large
  export BATCH_SIZE=16
  export NUM_EPOCHS=20
  export SAVE_STEPS=10000
  export TEXT_RESULT=test_result$j.txt
  export TEXT_PREDICTION=test_predictions$j.txt
  export SEED=$j

  export OUTPUT_DIR=baseline_models/bam_afroxlmr$j
  CUDA_VISIBLE_DEVICES=2 python3 ../train_pos.py --data_dir ../transfer_corpus/bm_crb/ \
  --model_type xlmroberta \
  --model_name_or_path $BERT_MODEL \
  --output_dir $OUTPUT_DIR \
  --test_result_file $TEXT_RESULT \
  --test_prediction_file $TEXT_PREDICTION \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --save_steps $SAVE_STEPS \
  --gradient_accumulation_steps 2 \
  --seed $SEED \
  --do_predict \
  --overwrite_output_dir


  export OUTPUT_DIR=baseline_models/pcm_afroxlmr$j
  CUDA_VISIBLE_DEVICES=2 python3 ../train_pos.py --data_dir ../transfer_corpus/pcm_nsc/ \
  --model_type xlmroberta \
  --model_name_or_path $BERT_MODEL \
  --output_dir $OUTPUT_DIR \
  --test_result_file $TEXT_RESULT \
  --test_prediction_file $TEXT_PREDICTION \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --save_steps $SAVE_STEPS \
  --gradient_accumulation_steps 2 \
  --seed $SEED \
  --do_predict \
  --overwrite_output_dir


  export OUTPUT_DIR=baseline_models/wol_afroxlmr$j
  CUDA_VISIBLE_DEVICES=2 python3 ../train_pos.py --data_dir ../transfer_corpus/wo_wtb/ \
  --model_type xlmroberta \
  --model_name_or_path $BERT_MODEL \
  --output_dir $OUTPUT_DIR \
  --test_result_file $TEXT_RESULT \
  --test_prediction_file $TEXT_PREDICTION \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --save_steps $SAVE_STEPS \
  --gradient_accumulation_steps 2 \
  --seed $SEED \
  --do_predict \
  --overwrite_output_dir


  export OUTPUT_DIR=baseline_models/yor_afroxlmr$j
  CUDA_VISIBLE_DEVICES=2 python3 ../train_pos.py --data_dir ../transfer_corpus/yo_ytb/ \
  --model_type xlmroberta \
  --model_name_or_path $BERT_MODEL \
  --output_dir $OUTPUT_DIR \
  --test_result_file $TEXT_RESULT \
  --test_prediction_file $TEXT_PREDICTION \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --save_steps $SAVE_STEPS \
  --gradient_accumulation_steps 2 \
  --seed $SEED \
  --do_predict \
  --overwrite_output_dir


done
