# output/DBpedia/N-Former/20220910_184258/
# output/DBpedia/N-Former/20220911_234444/

python nformer_train.py \
   --task validate \
   --model_path output/DBpedia/N-Former/20220911_234444/nformer.bin \
   --data_path datasets/DWY15K/DBpedia15K \
   --output_path output/DBpedia \
   --batch_size 2048 \
   --device cuda:1 \
   --num_workers 32 \
   --pin_memory True