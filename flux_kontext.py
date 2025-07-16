import torch
from diffusers import FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from transformers import T5EncoderModel
from diffusers.utils import load_image
from PIL import Image
import os

text_encoder_args = {
    "torch_dtype": torch.bfloat16, "local_files_only": True
}
text_encoder_id = "t5-encoder-fp16"
text_encoder_2 = T5EncoderModel.from_pretrained(
                text_encoder_id, **text_encoder_args)
transformer = FluxTransformer2DModel.from_single_file(
                "https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF/blob/main/flux1-kontext-dev-Q4_0.gguf",
                # quantization_config=quant_config,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16)
pipeline_args = {
    "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-Kontext-dev",
    "transformer": transformer,
    "text_encoder_2": text_encoder_2,
    "torch_dtype": torch.bfloat16,
    "local_files_only": False
}

pipe = FluxKontextPipeline.from_pretrained(**pipeline_args)
# pipe.to("cuda")
pipe.enable_model_cpu_offload()

left = Image.open("jjk_left.png")
right = Image.open("cp_right.png")
#
# # Resize to match height if needed
# left = left.resize((512, 512))
# right = right.resize((512, 512))
stitched = Image.new('RGB', (left.width + right.width, left.height))
stitched.paste(left, (0, 0))
stitched.paste(right, (left.width, 0))

image = pipe(
    image=stitched,
    # cond_image=[left, right],
    prompt="combine the images but only use the face of the left image pasting onto the person of the right image",
    guidance_scale=3.5,
    num_inference_steps=28,
).images[0]

image.show()
image.save("jjk_combined.png")