# positive samples
echo "=== Running positive samples ==="
python /root/fengyuan/tools/t2i_data_tools/t2i_gen/images_gen/images_gen_pipeline.py \
    --model_name "Z-Image-Turbo" \
    --data_file "/root/fengyuan/datasets/vision_auto_rubric/for_rubrics/positive/split_1.json" \
    --prompt_choice "prompt" \
    --postive "positive" 

# negative samples part1: text-image alignment (z-image-turbo)
echo "=== Running negative samples part1 ==="
python /root/fengyuan/tools/t2i_data_tools/t2i_gen/images_gen/images_gen_pipeline.py \
    --model_name "Z-Image-Turbo" \
    --data_file "/root/fengyuan/datasets/vision_auto_rubric/for_rubrics/negative/split_1.json" \
    --prompt_choice "prompt_fault" \
    --postive "negative" 

# negative samples part2: flux.1 dev 
echo "=== Running negative samples part2 ==="
python /root/fengyuan/tools/t2i_data_tools/t2i_gen/images_gen/images_gen_pipeline.py \
    --model_name "FLUX.1-dev" \
    --data_file "/root/fengyuan/datasets/vision_auto_rubric/for_rubrics/negative/split_2.json" \
    --prompt_choice "prompt" \
    --postive "negative" 

# negative samples part3: sd3.5-medium
echo "=== Running negative samples part3 ==="
python /root/fengyuan/tools/t2i_data_tools/t2i_gen/images_gen/images_gen_pipeline.py \
    --model_name "SD3.5-medium" \
    --data_file "/root/fengyuan/datasets/vision_auto_rubric/for_rubrics/negative/split_3.json" \
    --prompt_choice "prompt" \
    --postive "negative" 
   
# negative samples part4: sana1.5_4.8b
echo "=== Running negative samples part4 ==="
python /root/fengyuan/tools/t2i_data_tools/t2i_gen/images_gen/images_gen_pipeline.py \
    --model_name "SANA1.5_4.8B" \
    --data_file "/root/fengyuan/datasets/vision_auto_rubric/for_rubrics/negative/split_4.json" \
    --prompt_choice "prompt" \
    --postive "negative" 
