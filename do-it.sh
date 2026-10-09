for t in `seq -w 1 50`; do
	echo AAAAAAA Topic cd2026_0$t AAAAAAAA
	topic-judges train --encoder-type bi --sentence-transformer --lora     --model-name joe32140/ModernBERT-base-msmarco     --train-data data/external/train-cdet-hard-50-50.parquet     --query-id cd2026_0$t     --max-length 384 --max-length-doc 2048     --batch-size 8 --accumulate-batches 4     --output-dir ./outputs --device cuda 
done
