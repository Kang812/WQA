from unsloth import FastVisionModel # FastLanguageModel for LLMs

def load_model(model_name, load_in_4bit, use_gradient_checkpointing):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit = load_in_4bit, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = use_gradient_checkpointing, # True or "unsloth" for long context
    )
    return model, tokenizer
