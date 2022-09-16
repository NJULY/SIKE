python nformer_joint_train.py \
   --task validate \
   --model_path output/DWY15K/N-Former/20220915_222650/DBpedia15K_nformer.bin \
   --data_path datasets/DWY15K \
   --output_path output/DWY15K \
   --batch_size 2048 \
   --device cuda:3 \
   --num_workers 32 \
   --pin_memory True