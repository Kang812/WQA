import cv2
import albumentations as A
import pandas as pd
from tqdm import tqdm
from PIL import Image
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
import argparse
import sys
sys.path.append("/workspace/whole_slide_image_LLM/wsi_level_vqa-main")
from models.vqa_model import load_model
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from dataset.vqa_dataset import convert_dataset

def main(args):
    transform = A.Compose([
        A.LongestMaxSize(512),
        A.PadIfNeeded(
            min_height=512,
            min_width=512,
            border_mode=cv2.BORDER_CONSTANT,
            value=(255, 255, 255)  # RGB 흰색
        )])
    
    model, tokenizer = load_model(args.model_name, args.load_in_4bit, args.use_gradient_checkpointing)

    train_df = pd.read_csv(args.train_df_path)

    print("Train image data convert")
    train_converted_data = convert_dataset(train_df, transform)

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = False, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers

        r = args.r,           # The larger, the higher the accuracy, but might overfit
        lora_alpha = args.lora_alpha,  # Recommended alpha == r at least
        lora_dropout = args.lora_dropout,
        bias = "none",
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None,)
    
    FastVisionModel.for_training(model) # Enable for training!
    
    trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
            train_dataset = train_converted_data,
            args = SFTConfig(
                per_device_train_batch_size = args.per_device_train_batch_size,
                gradient_accumulation_steps = args.gradient_accumulation_steps,
                warmup_steps = 5,
                #max_steps = ((len(train_converted_data)/args.per_device_train_batch_size) * args.max_steps),
                num_train_epochs = args.num_train_epochs, # Set this instead of max_steps for full training runs
                learning_rate = args.lr,
                fp16 = not is_bf16_supported(),
                bf16 = is_bf16_supported(),
                logging_steps = 1,
                optim = args.optim,
                weight_decay = args.weight_decay,
                lr_scheduler_type = args.lr_scheduler_type,
                seed = 3407,
                output_dir = args.output_dir,
                report_to = "none",     # For Weights and Biases

                # You MUST put the below items for vision finetuning:
                remove_unused_columns = False,
                dataset_text_field = "",
                dataset_kwargs = {"skip_prepare_dataset": True},
                dataset_num_proc = 4,
                max_seq_length = args.max_seq_length,
            ),
        )
    
    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model loaded !!
    parser.add_argument('--model_name', type=str, default='unsloth/Llama-3.2-11B-Vision-Instruct')
    parser.add_argument('--load_in_4bit', type=bool, default=True)
    parser.add_argument('--use_gradient_checkpointing', type = str, default='unsloth')
    
    # train dataframe
    parser.add_argument('--train_df_path', type = str, default='/workspace/whole_slide_image_LLM/data/vqa_dataset/vqa_train.csv')

    # lora
    parser.add_argument('--r', type = int, default=16)
    parser.add_argument('--lora_alpha', type = int, default=16)
    parser.add_argument('--lora_dropout', type = int, default=0)

    # trainer
    parser.add_argument('--per_device_train_batch_size', type = int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type = int, default=4)
    parser.add_argument('--warmup_steps', type = int, default=5)
    parser.add_argument('--num_train_epochs', type = int, default=2)
    parser.add_argument('--lr', type = float, default=2e-4)
    parser.add_argument('--optim', type = str, default="adamw_8bit")
    parser.add_argument('--weight_decay', type = float, default=0.01)
    parser.add_argument('--lr_scheduler_type', type = str, default="linear")
    parser.add_argument('--output_dir', type = str, default='/workspace/whole_slide_image_LLM/data/vqa_dataset/save_path/')
    parser.add_argument('--max_seq_length', type = int, default = 2048)
    
    args = parser.parse_args()
    print("Argument:")
    for k, v in args.__dict__.items():
        print(f' {k}: {v}')
    
    print()
    main(args)