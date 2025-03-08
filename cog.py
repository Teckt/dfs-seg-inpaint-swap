import os
import subprocess
import time

import PIL.Image as Image
import numpy as np
import cv2
import torch
from diffusers import CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel, GGUFQuantizationConfig
from diffusers.utils.export_utils import export_to_video, _legacy_export_to_video
from tqdm import tqdm
from transformers import T5EncoderModel

from redresser_utils import RedresserSettings
from optimum.quanto import quantize, freeze, qfloat8


def run_ffmpeg_optical_flow(input_video: str, output_video: str, fps: int = 60):
    """
    Run FFmpeg's minterpolate filter to calculate optical flow and interpolate frames.

    Args:
        input_video (str): Path to the input video file.
        output_video (str): Path to the output video file.
        fps (int): The target frames per second (default is 60).
    """
    # Define the FFmpeg command with minterpolate filter
    if "gif" in output_video[:-4]:
        ffmpeg_command = [
            'ffmpeg',
            '-y',  # Overwrite the output file if it exists
            '-i', input_video,  # Input video
            '-vf', f"minterpolate='fps={fps}'",  # Video filter for optical flow
            # '-preset', 'veryslow',  # Use slower, better compression for quality
            # '-crf', '17',  # Constant Rate Factor, lower is better quality (18-23 is good)
            output_video  # Output video file
        ]
    else:
        ffmpeg_command = [
            'ffmpeg',
            '-y',  # Overwrite the output file if it exists
            '-i', input_video,  # Input video
            '-vf', f"minterpolate='fps={fps}'",  # Video filter for optical flow
            '-c:v', 'libx264',  # Use H.264 codec for video
            '-b:v', '5000k',  # Set bitrate to 5000 kbps (adjust as needed)
            '-preset', 'veryslow',  # Use slower, better compression for quality
            '-crf', '17',  # Constant Rate Factor, lower is better quality (18-23 is good)
            output_video  # Output video file
        ]

    try:
        # Run the FFmpeg command
        result = subprocess.run(ffmpeg_command, check=True, text=True, capture_output=True)
        print("FFmpeg Output:", result.stdout)
        print("FFmpeg Error (if any):", result.stderr)
        print(f"Successfully processed video: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg command failed with error: {e.stderr}")


class CogSettings(RedresserSettings):

    default_options = {
        "prompt": "",
        "image": "images",
        "use_dynamic_cfg": False,
        "guidance_scale": 6.0,
        "num_inference_steps": 20,
        "num_frames": 49,
        "seed": -1,
    }

    def __init__(self):
        super().__init__()
        self.previous_options = CogSettings.default_options.copy()


class VideoGenerator:
    def __init__(self, is_server=False, local_files_only=True, model="cog-i2v"):
        '''

        :param is_server: determines the output name in redresser_output_file_path
        :param local_files_only:
        '''
        self.is_server = is_server
        self.model = model
        # initialize default settings first
        self.settings = CogSettings()

        self.dtype = torch.bfloat16
        self.flux_model_id = "black-forest-labs/FLUX.1-dev"  # Flux dev
        self.cog_model_id = "THUDM/CogVideoX-5b-I2V"  # Cog 5b I2V
        # self.cog_model_id = "D:/huggingface/models--THUDM--CogVideoX-5b-I2V/snapshots/c5c783ca1606069b9996dc56f207cc2e681691ed"
        with tqdm(range(8), "preparing cog pipeline") as p:
            p.set_description("Loading text encoder 2")
            self.text_encoder_2 = T5EncoderModel.from_pretrained(self.flux_model_id, subfolder="text_encoder_2",
                                                                 torch_dtype=self.dtype, local_files_only=True)
            p.update()
            p.set_description("Loading cog pipeline")
            self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                self.cog_model_id,
                text_encoder=self.text_encoder_2,
                torch_dtype=self.dtype, local_files_only=True)
            p.update()
            # p.set_description("Loading cog transformer")
            # self.transformer = CogVideoXTransformer3DModel.from_single_file(
            #     'https://huggingface.co/Kijai/CogVideoX_GGUF/resolve/main/CogVideoX_5b_1_5_I2V_GGUF_Q4_0.safetensors',
            #     quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
            #     torch_dtype=self.dtype,
            # )
            # self.pipe.transformer = self.transformer
            # p.update()

            # quantize cog transformer
            p.desc = "quantize cog transformer"
            quantize(self.pipe.text_encoder, weights=qfloat8)
            p.update()
            p.desc = "freeze cog transformer"
            freeze(self.pipe.text_encoder)
            p.update()
            # quantize cog transformer
            p.desc = "quantize cog transformer"
            quantize(self.pipe.transformer, weights=qfloat8)
            p.update()
            p.desc = "freeze cog transformer"
            freeze(self.pipe.transformer)
            p.update()
            # quantize cogx vae
            p.desc = "quantize cog vae"
            quantize(self.pipe.vae, weights=qfloat8)
            p.update()
            p.desc = "freeze cog vae"
            freeze(self.pipe.vae)
            p.update()

        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

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

        args = {
            "prompt": prompt,
            "image": image,
            # "height": self.settings.options["height"],
            # "width": self.settings.options["width"],
            "use_dynamic_cfg": self.settings.options["use_dynamic_cfg"],
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
        image = np.array(image)
        h, w = image.shape[:2]
        max_h = 480
        max_w = 720

        if h > w:  # vertical
            ratio = 480 / 720
            target_h = 480
            target_w = int(ratio * target_h)
            image = cv2.resize(image, (target_w, target_h))
        elif w > h:  # horizontal
            ratio = 720 / 480
            target_w = 720
            target_h = int(ratio * target_w)
            image = cv2.resize(image, (target_w, target_h))
        else:  # square
            image = cv2.resize(image, (max_h, max_h))

        ratio = 720 / 480
        target_w = int(ratio * h)
        cog_image = np.zeros(shape=(h, target_w, 3), dtype=np.uint8)
        center_x, center_y = target_w / 2, h / 2
        x1 = int(center_x - w / 2)
        x2 = x1 + w
        y1 = 0
        y2 = h
        cog_image[y1:y2, x1:x2, :] = image
        return Image.fromarray(cog_image)


