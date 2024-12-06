# text to 3d fast
python main.py \
    --text_prompt "Blocky text spelling \"Dog\"" \
    --save_folder ./outputs/test/ \
    --text2image_path weights/flux1dev \
    --max_faces_num 10000 \
    --use_lite