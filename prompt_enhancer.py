import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model checkpoint
model_checkpoint = "gokaygokay/Flux-Prompt-Enhance"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Model
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

enhancer = pipeline('text2text-generation',
                    model=model,
                    tokenizer=tokenizer,
                    repetition_penalty= 1.2,
                    device=device)

max_target_length = 512
prefix = "enhance prompt: "

short_prompt = "jjk"
answer = enhancer(prefix + short_prompt, max_length=max_target_length)
final_answer = answer[0]['generated_text']
print(final_answer)