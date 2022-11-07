export MAX_LENGTH=150
export BERT_MODEL=bert-base-multilingual-cased

#export MAX_LENGTH=150
python3 preprocess.py data/pos_data/yor.txt $BERT_MODEL $MAX_LENGTH > data/pos_data_length_adjusted/yor.txt


