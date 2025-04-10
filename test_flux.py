import cv2
import torch
from diffusers import FluxPipeline, FluxPriorReduxPipeline, FluxTransformer2DModel
from tqdm import tqdm
from transformers import T5EncoderModel, CLIPTextModel
import PIL.Image as Image

from CONSTANTS import *

dtype = torch.bfloat16
# load tokenizer
with tqdm(range(1), "Loading CLIPTextModel") as progress_bar:
    clip_L_text_encoder = CLIPTextModel.from_pretrained(
        TEXT_ENCODER_CLIP_L_PATH,
        torch_dtype=dtype,
        local_files_only=USE_LOCAL_FILES)
    progress_bar.update()

# check if model name is Anyfusion and exists locally without the repo id
text_encoder_args = {
    "torch_dtype": dtype, "local_files_only": USE_LOCAL_FILES
}
if os.path.exists(os.path.join("t5-encoder-fp16", "model.safetensors")):
    text_encoder_id = "t5-encoder-fp16"
else:
    text_encoder_id = FLUX_PATH
    text_encoder_args["subfolder"] = "text_encoder_2"
with tqdm(total=1, desc="loading text_encoder_2") as progress_bar:
    text_encoder_2 = T5EncoderModel.from_pretrained(
        text_encoder_id, **text_encoder_args)
    progress_bar.update()

transformer_args = {
    "torch_dtype": dtype, "local_files_only": USE_LOCAL_FILES
}
transformer = FluxTransformer2DModel.from_single_file("C:\\Users\\teckt\\.cache\\huggingface\\hub\\models--Kijai--flux-fp8\\snapshots\\e77f550e3fe8be226884d5944a40abdbe4735ff5\\flux1-dev-fp8.safetensors",
    subfolder="transformer", **transformer_args)
pipeline_args = {
    "pretrained_model_name_or_path": FLUX_PATH,
    "transformer": transformer,
    "text_encoder": clip_L_text_encoder,
    "text_encoder_2": text_encoder_2,
    "torch_dtype": dtype,
    "local_files_only": USE_LOCAL_FILES
}
pipe = FluxPipeline.from_pretrained(

    **pipeline_args)
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

pipe.load_lora_weights(
    LORA_PATH,
    weight_name=f"jjk-d-step00000300.safetensors",
    adapter_name="jjk")
pipe.load_lora_weights(
    LORA_PATH,
    weight_name=f"jjk-d-step00000300.safetensors",
    adapter_name="jjk")
lora_settings = {
                "adapter_names": ["jjk"],
                "adapter_weights": [1.0]
            }
pipe.set_adapters(**lora_settings)

prompt = "close-up shot of a jjk face, shoulder-length hair"
for i in range(4):
    out = pipe(
        prompt=prompt,
        guidance_scale=3.5,
        height=1024,
        width=1024,
        num_inference_steps=20,
    ).images[0]
    out.save(f"flux-test_{i}.png")