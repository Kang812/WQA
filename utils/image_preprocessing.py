import numpy as np
import cv2
import os
import argparse
from glob import glob
from tqdm import tqdm
from mmseg.apis import init_model, inference_model

def crop_image(image, mask):
    for w_pos in reversed(range(image.shape[1])):
        if (mask[:, w_pos] == [0]).all():
                image = np.delete(image, w_pos, 1)
    
    for h_pos in reversed(range(image.shape[0])):
        if (mask[h_pos, :] == [0]).all():
            image = np.delete(image, h_pos, 0)

    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='/workspace/whole_slide_image_LLM/wsi_level_vqa-main/utils/wsi_core/segmentation/pidnet_cfg.py')
    parser.add_argument('--model_ckpt', type=str, default='/workspace/whole_slide_image_LLM/wsi_level_vqa-main/utils/wsi_core/segmentation/best_mIoU_iter_6400.pth')
    parser.add_argument('--data_dir', type = str, default='/workspace/whole_slide_image_LLM/data/train_imgs/')
    parser.add_argument('--save_dir', type = str, default='/workspace/whole_slide_image_LLM/data/vqa_dataset/wsi_image/')
    args = parser.parse_args()

    print("Tissue Detect Model Loaded!!")
    config = args.model_config
    checkpoints = args.model_ckpt
    model = init_model(config, checkpoints)

    image_paths = glob(os.path.join(args.data_dir, "*.png"))

    for i in tqdm(range(len(image_paths))):
        image_path = image_paths[i]
        file_name = image_path.split("/")[-1]
        img = cv2.imread(image_path)
        result = inference_model(model, img)
        mask = result.pred_sem_seg.data.detach().cpu().numpy()[0]
        img = crop_image(img, mask)
        cv2.imwrite(os.path.join(args.save_dir, file_name), img)
