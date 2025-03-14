import os
import time

import numpy as np
import PIL.Image as Image
import cv2
import torch
import tqdm
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan, WanTransformer3DModel, GGUFQuantizationConfig, \
    UniPCMultistepScheduler

from diffusers.utils.export_utils import _legacy_export_to_video
from tqdm import tqdm
from transformers import CLIPVisionModel, T5EncoderModel, T5Tokenizer, UMT5EncoderModel, \
    T5TokenizerFast
from diffusers import BitsAndBytesConfig

from cog import run_ffmpeg_optical_flow, VideoGenerator
from redresser_utils import RedresserSettings


class WanSettings(RedresserSettings):

    default_options = {
        "prompt": "",
        "image": "images/1741761675-3.5-20-outputImage.png",
        "max_area": 65536,
        "guidance_scale": 5.0,
        "num_inference_steps": 30,
        "num_frames": 49,
        "seed": -1,
    }

    def __init__(self):
        super().__init__()
        self.previous_options = WanSettings.default_options.copy()


class WanVideoGenerator(VideoGenerator):
    def __init__(self, is_server=False, local_files_only=True, model="wan-480"):
        '''

        :param is_server: determines the output name in redresser_output_file_path
        :param local_files_only:
        '''
        super().__init__(is_server, local_files_only, model)
        self.is_server = is_server
        self.model = model
        # initialize default settings first
        self.settings = WanSettings()

        self.dtype = torch.bfloat16

        # self.load_quanto_pipe()
        self.pipe = None
        self.load_pipe()

    def load_pipe(self):
        model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

        quant_config = BitsAndBytesConfig(load_in_4bit=True)

        with tqdm(desc="Loading tokenizer"):
            tokenizer = T5TokenizerFast.from_pretrained(
                model_id,
                # quantization_config=quant_config,
                subfolder="tokenizer",
                torch_dtype=torch.bfloat16
            )
        # with tqdm(desc="Saving tokenizer"):
        #     tokenizer.save_pretrained("umt5-xxl-tokenizer-nf4", max_shard_size="16GB")

        with tqdm(desc="Loading vae"):
            vae = AutoencoderKLWan.from_pretrained(
                model_id, subfolder="vae", torch_dtype=torch.bfloat16)

        with tqdm(desc="Loading image_encoder"):
            image_encoder = CLIPVisionModel.from_pretrained(
                model_id, subfolder="image_encoder",
                # quantization_config=quant_config,
                torch_dtype=torch.bfloat16
            )
        # with tqdm(desc="Saving image_encoder"):
        #     image_encoder.save_pretrained("clip-vision-nf4", max_shard_size="16GB")

        text_encoder_path = "Anyfusion/umt5-xxl-encoder-fp8"
        if "Anyfusion/" in text_encoder_path:
            local_dir = text_encoder_path.replace("Anyfusion/", "")
            if os.path.exists(os.path.join(local_dir, "diffusion_pytorch_model.safetensors")):
                text_encoder_path = local_dir
        with tqdm(desc="Loading text_encoder"):
            text_encoder = UMT5EncoderModel.from_pretrained(
                text_encoder_path,
                # model_id,
                # subfolder="text_encoder",
                # quantization_config=quant_config,

                torch_dtype=torch.bfloat16
            )
            # text_encoder.to("cuda", torch.float8_e4m3fn)
        # with tqdm(desc="Saving text_encoder"):
        #     text_encoder.save_pretrained("umt5-xxl-encoder-fp8", max_shard_size="16GB")
        # exit()
        with tqdm(desc="Loading transformer"):
            transformer = WanTransformer3DModel.from_single_file(
                "https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/wan2.1-i2v-14b-480p-Q4_0.gguf",
                # quantization_config=quant_config,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16)

        self.pipe = WanImageToVideoPipeline.from_pretrained(
            model_id,
            text_encoder=text_encoder, tokenizer=tokenizer,
            transformer=transformer, vae=vae, image_encoder=image_encoder,
            torch_dtype=torch.bfloat16
        )

        flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=flow_shift)

        self.pipe.enable_model_cpu_offload()
        # pipe.vae.enable_slicing()

        # pipe.to("cuda")
    def run(self):
        image_path = self.settings.options.get("image")

        seed = self.settings.options.get("seed", -1)
        if seed is None or seed < 0:
            seed = int(time.time())

        print("seed", seed)
        generator = torch.Generator().manual_seed(seed)

        # filter and extract loras from prompt and apply them to the pipe
        prompt = self.settings.options["prompt"]
        if self.model == "flux":
            prompt = self.pipe.apply_flux_loras_with_prompt(prompt, use_turbo=True)

        image = self.load_and_prepare_image(image_path)
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        args = {
            "prompt": prompt,
            "negative_prompt":negative_prompt,
            "image": image,
            "height": image.height,
            "width": image.width,

            "num_frames": self.settings.options["num_frames"],
            "guidance_scale": self.settings.options["guidance_scale"],
            "num_inference_steps": self.settings.options["num_inference_steps"],
            # "guidance_scale": random.uniform(3.5, 7.5),  # self.settings.options["guidance_scale"],
            # "num_inference_steps": 8  # self.settings.options["num_inference_steps"],
            # "max_sequence_length": 512,
            # "generator": torch.Generator("cpu").manual_seed(88)
        }

        # if self.settings.options["negative_prompt"] is not None:
        #     args["negative_prompt"] = self.settings.options["negative_prompt"]
        # if self.settings.options["strength"] is not None:
        #     args["strength"] = self.settings.options["strength"]
        # if self.settings.options["clip_skip"] > 0:
        #     args["clip_skip"] = self.settings.options["clip_skip"]
        if generator is not None:
            args["generator"] = generator

        video = self.pipe(
            **args,
        )

        if self.is_server:
            basename = os.path.basename(image_path)
            redresser_dir = image_path.replace(basename, "")
            input_file_name = f"{redresser_dir}/outputVideo.mp4"
        else:
            # save to same dir as image
            redresser_slug = "video_results"
            if os.path.exists(image_path):
                basename = os.path.basename(image_path)

                redresser_dir = image_path.replace(basename, redresser_slug)
            else:
                basename = "outputImage.png"
                redresser_dir = redresser_slug
            input_file_name = f"{redresser_dir}/{seed}_{os.path.basename(image_path)}_{self.settings.options['guidance_scale']}_{self.settings.options['num_inference_steps']}.mp4"
        if not os.path.exists(redresser_dir):
            os.mkdir(redresser_dir)
        output_file_name = f"{input_file_name}-opt.mp4"

        try:
            _legacy_export_to_video(video.frames[0], input_file_name, fps=8)
            run_ffmpeg_optical_flow(input_video=input_file_name, output_video=output_file_name)

        except Exception as e:
            print(e)
            result_dir = f"{redresser_dir}/{seed}"
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)
            for f_idx, frame in enumerate(video.frames[0]):

                frame.save(f"{result_dir}/{str(f_idx).zfill(5)}.png")

    def load_and_prepare_image(self, img_path):

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        # max_area = 480 * 832
        max_area = self.settings.options.get("max_area", 512*512)
        aspect_ratio = image.height / image.width
        mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))
        return image

