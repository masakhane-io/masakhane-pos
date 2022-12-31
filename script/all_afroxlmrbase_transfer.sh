for SRC_LANG in wo_wtb af_afribooms ar_padt en_ewt fr_gsd pcm_nsc ro_rrt eng-ron-wol
do
	for j in 1 2 3 4 5
	do
		export MAX_LENGTH=200
		export BERT_MODEL=Davlan/afro-xlmr-base
		export OUTPUT_DIR=transfer_models/${SRC_LANG}_afroxlmrbase$j
		export TEXT_RESULT=test_result$j.txt
		export TEXT_PREDICTION=test_predictions$j.txt
		export BATCH_SIZE=16
		export NUM_EPOCHS=20
		export SAVE_STEPS=10000
		export SEED=$j

		CUDA_VISIBLE_DEVICES=3 python3 ../train_pos.py --data_dir ../transfer_corpus/${SRC_LANG}/ \
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
		--do_train \
		--do_eval \
		--do_predict \
		--overwrite_output_dir

		for LANG in bam bbj ewe fon hau ibo kin lug luo mos nya pcm sna swa twi wol xho yor zul
		do
      export MAX_LENGTH=200
      export BERT_MODEL=Davlan/afro-xlmr-base
      export OUTPUT_DIR=transfer_models/${SRC_LANG}_afroxlmrbase$j
      export TEXT_RESULT=test_result_${SRC_LANG}_${LANG}.txt
      export TEXT_PREDICTION=test_predictions_${SRC_LANG}_${LANG}.txt
      export BATCH_SIZE=16
      export NUM_EPOCHS=20
      export SAVE_STEPS=10000
      export SEED=$j

      CUDA_VISIBLE_DEVICES=3 python3 ../train_pos.py --data_dir ../data/${LANG}/ \
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
	done
done





for SRC_LANG in wo_wtb af_afribooms ar_padt en_ewt fr_gsd pcm_nsc ro_rrt eng-ron-wol
do
	for j in 1 2 3 4 5
	do
		export MAX_LENGTH=200
		export BERT_MODEL=Davlan/afro-xlmr-base
		export OUTPUT_DIR=transfer_models/${SRC_LANG}_afroxlmrbase$j
		export TEXT_RESULT=test_result$j.txt
		export TEXT_PREDICTION=test_predictions$j.txt
		export BATCH_SIZE=16
		export NUM_EPOCHS=20
		export SAVE_STEPS=10000
		export SEED=$j

		for LANG in mos
		do
      export MAX_LENGTH=200
      export BERT_MODEL=Davlan/afro-xlmr-base
      export OUTPUT_DIR=transfer_models/${SRC_LANG}_afroxlmrbase$j
      export TEXT_RESULT=test_result_${SRC_LANG}_${LANG}.txt
      export TEXT_PREDICTION=test_predictions_${SRC_LANG}_${LANG}.txt
      export BATCH_SIZE=16
      export NUM_EPOCHS=20
      export SAVE_STEPS=10000
      export SEED=$j

      CUDA_VISIBLE_DEVICES=3 python3 ../train_pos.py --data_dir ../data/${LANG}/ \
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
	done
done
