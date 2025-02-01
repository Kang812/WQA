import base64
import cv2
import os
import json
import numpy as np
from tqdm import tqdm
from glob import glob

def get_mask(image_path, save_dir):
    
    img = cv2.imread(image_path)
    img = cv2.medianBlur(img, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img)
    t, t_otsu = cv2.threshold(s, -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = np.ones((7,7),np.uint8)
    get_mask = cv2.dilate(t_otsu, kernel,iterations = 2)

    contours, _  = cv2.findContours(get_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((img.shape[:2]))

    for contour in contours:
        if len(contour) > 100:
            mask = cv2.fillPoly(mask, [contour], 1)
    
    file_name = image_path.split("/")[-1]
    name, ext = file_name.split(".")

    img = cv2.imread(image_path)
    cv2.imwrite(os.path.join(save_dir, 'images', file_name), img)
    cv2.imwrite(os.path.join(save_dir, "masks", file_name), mask)


if __name__ == '__main__':
    #image_path = '/workspace/whole_slide_image_LLM/data/train_imgs/BC_01_3048.png'
    #save_dir = '/workspace/whole_slide_image_LLM/wsi_level_vqa-main/'
    #get_mask(image_path ,save_dir)

    image_paths = glob("/workspace/whole_slide_image_LLM/data/train_imgs/*.png")
    save_dir = '/workspace/whole_slide_image_LLM/data/semantic_segmentation_dataset/'
    for i in tqdm(range(len(image_paths))):
        image_path = image_paths[i]
        get_mask(image_path, save_dir)