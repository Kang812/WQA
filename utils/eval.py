import evaluate
import torch
import numpy as np
import argparse
import cv2
import pandas as pd
from PIL import Image
import albumentations as A
from unsloth import FastVisionModel # FastLanguageModel for LLMs
from tqdm import tqdm
import torch

class metrics:
    def __init__(self, tokenizer):
        super().__init__()
        self.blue = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.meteor = evaluate.load("meteor")
        self.tokenizer = tokenizer

    def compute_metrics(self, pred , real):
        bleu = self.blue.compute(predictions=pred, references=real)
        rouge = self.rouge.compute(predictions=pred, references=real)
        meteor = self.meteor.compute(predictions=pred, references=real)

        return {
            "bleu": bleu["bleu"],
            **rouge,
            **meteor,}

def main(args):
    transform = A.Compose([
        A.LongestMaxSize(512),
        A.PadIfNeeded(
            min_height=512,
            min_width=512,
            border_mode=cv2.BORDER_CONSTANT,
            value=(255, 255, 255)  # RGB 흰색
        )])
    
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_path,
        load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
        )
    
    FastVisionModel.for_inference(model)

    df = pd.read_csv(args.df_path)
    compute_met = metrics(tokenizer).compute_metrics
    
    total_dict = dict()
    total_dict['bleu'] = 0
    total_dict['rouge1'] = 0
    total_dict['rouge2'] = 0 
    total_dict['rougeL'] = 0
    total_dict['rougeLsum'] = 0 
    total_dict['meteor'] = 0

    for i in tqdm(range(df.shape[0])):
        image_path = df.iloc[i]['wsi_image']
        instruction = df.iloc[i]['q']
        real = df.iloc[i]['a']

        image = cv2.imread(image_path)
        image = transform(image = image)['image']
        image = Image.fromarray(image)
        
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}]}]

        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
        inputs = tokenizer( image, input_text, add_special_tokens = False, return_tensors = "pt",).to("cuda")
        predict_token = model.generate(**inputs, max_new_tokens = 400, use_cache = True, temperature = 1.5, min_p = 0.1)
        predict = tokenizer.batch_decode(predict_token)[0].split("assistant")[1].replace("<|end_header_id|>\n\n", "").replace("<|eot_id|>", "")
        
        res = compute_met([predict], [real])
        total_dict["bleu"] += res['bleu']
        total_dict["rouge1"] += res['rouge1']
        total_dict["rouge2"] += res['rouge2']
        total_dict["rougeL"] += res['rougeL']
        total_dict["rougeLsum"] += res['rougeLsum']
        total_dict["meteor"] += res['meteor']
    
    for k in list(total_dict.keys()):
        total_dict[k] = total_dict[k]/df.shape[0]
    print(total_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_path', type = str, default='/workspace/whole_slide_image_LLM/data/vqa_dataset/vqa_test.csv')
    parser.add_argument('--model_path' , type = str, default='/workspace/whole_slide_image_LLM/data/vqa_dataset/save_path/checkpoint-2940/')

    args = parser.parse_args()
    main(args)
    # {'bleu': 0.678539334353433, 'rouge1': 0.7329251523084669, 'rouge2': 0.5846770038455502, 'rougeL': 0.7326162088009939, 'rougeLsum': 0.7327104772776152, 'meteor': 0.7878066707704559}