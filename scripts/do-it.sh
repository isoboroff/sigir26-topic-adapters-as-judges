for qid in `seq -w 2 50`; do
    echo Query cd2026_0${qid}
    python train_cdet.py  --train_data_path ../data/external/train-cdet.parquet --predict_data_path ../data/external/test-cdet.parquet --query_id cd2026_0${qid} --device cuda --batch_size 16 --max_length 512 --accumulate_batches 1 --lora True
done
