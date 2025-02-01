python ./utils/image_preprocessing.py --model_config /workspace/whole_slide_image_LLM/wsi_level_vqa-main/utils/wsi_core/segmentation/pidnet_cfg.py \
    --model_ckpt /workspace/whole_slide_image_LLM/wsi_level_vqa-main/utils/wsi_core/segmentation/best_mIoU_iter_6400.pth \
    --data_dir /workspace/whole_slide_image_LLM/data/train_imgs/ \
    --save_dir /workspace/whole_slide_image_LLM/data/vqa_dataset/wsi_image/