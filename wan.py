import os
import time
import PIL
import huggingface_hub
import numpy as np
import PIL.Image as Image
import cv2
import torch
import tqdm
from PIL import ImageFile, ImageOps
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan, WanTransformer3DModel, GGUFQuantizationConfig, \
    UniPCMultistepScheduler, WanVACEPipeline, WanVideoToVideoPipeline, WanVACETransformer3DModel
from controlnet_aux.processor import Processor
from diffusers.hooks import apply_group_offloading
from diffusers.utils.export_utils import _legacy_export_to_video
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import CLIPVisionModel, T5EncoderModel, T5Tokenizer, UMT5EncoderModel, \
    T5TokenizerFast
from diffusers import BitsAndBytesConfig
from ultralytics import YOLO

from cog import run_ffmpeg_optical_flow, VideoGenerator
from tf_free_functions import paste_swapped_image, paste_image_with_mask
from redresser_utils import RedresserSettings, yolo8_extract_faces, ImageResizeParams, yolo_segment_image
from CONSTANTS import *
from repainter_image_processor import ImageProcessor, expand_mask


def concatenate_videos(video_path1, video_path2, output_path):
    clip1 = moviepy.VideoFileClip(video_path1)
    clip2 = moviepy.VideoFileClip(video_path2)
    final_clip = moviepy.concatenate_videoclips([clip1, clip2])
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    clip1.close()
    clip2.close()
    final_clip.close()


def apply_flux_loras_with_prompt(pipe, prompt):
    # use_turbo = False
    loaded_loras = {}
    pipe.unload_lora_weights()
    # if use_turbo:
    #     prompt += "<turbo-a>"
    filtered_prompt, loras = extract_lora_params_from_prompt(prompt)
    if len(loras) == 0:
        # set
        active_adapters = pipe.get_active_adapters()
        print("no loras detected in prompt. setting adapters:", active_adapters)
        return filtered_prompt, {}

    # return filtered_prompt

    loras_filtered_names = {}
    # filter the names for periods or will throw an error when loading the lora
    for adapter_name, adapter_scale in loras.items():
        adapter_name_filtered_for_periods = adapter_name.replace(".", "dot")
        adapter_name_filtered_for_periods = adapter_name_filtered_for_periods.replace("/", "slash")
        loras_filtered_names[adapter_name] = adapter_name_filtered_for_periods

    lora_path = LORA_PATH
    for adapter_name, adapter_scale in loras.items():
        adapter_name_filtered_for_periods = loras_filtered_names[adapter_name]
        # load the lora into pipeline
        lora_file = os.path.join(lora_path, f"{adapter_name}.safetensors")
        if os.path.exists(lora_file):
            pipe.load_lora_weights(lora_path,
                                        weight_name=f"{adapter_name}.safetensors",
                                        adapter_name=adapter_name_filtered_for_periods)
            print("Loaded lora into pipe from", lora_file)
        else:
            # try to download from huggingface
            try:
                pipe.load_lora_weights(
                    "Anyfusion/flux-loras",
                    weight_name=f"{adapter_name}.safetensors",
                    adapter_name=adapter_name_filtered_for_periods)
                print("Loaded lora into pipe from", lora_file)
            except huggingface_hub.errors.RepositoryNotFoundError:
                print("WARNING:", f"{adapter_name}.safetensors", adapter_name_filtered_for_periods, "does not exist.",
                      "Ignoring.")
                continue
            except Exception as e:
                print("WARNING:", str(e), "Ignoring.")
                continue

        loaded_loras[adapter_name_filtered_for_periods] = float(adapter_scale)
    if len(loaded_loras) > 0:
        lora_settings = {
            "adapter_names": list(loaded_loras.keys()),
            "adapter_weights": list(loaded_loras.values())
        }
        pipe.set_adapters(**lora_settings)
        active_adapters = pipe.get_active_adapters()
        print("loras set:", active_adapters)
    else:
        print("loras set:", None)
    return filtered_prompt, loaded_loras


def extract_lora_params_from_prompt(prompt):
    assert isinstance(prompt, str)

    loras = {}
    open_bracket, closed_bracket = "<", ">"
    prompt_dict = {"prompt": prompt, "current_prompt": prompt}
    eos = False
    inside_brackets = False
    open_bracket_idx = 0
    closed_bracket_idx = 0
    while True:
        if len(prompt_dict["current_prompt"]) == 0:
            break
        for idx, char in enumerate(prompt_dict["current_prompt"]):
            if not inside_brackets:
                # start looking for the first open bracket you come across
                if char == open_bracket:
                    open_bracket_idx = idx
                    inside_brackets = True
                    continue  # go to next loop right away
            else:
                # start looking for the first closed bracket you come across
                if char == closed_bracket:
                    closed_bracket_idx = idx
                    inside_brackets = False
                    # create the lora params inside the brackets if not empty
                    adapter_name, adapter_scale = create_lora_params_from_prompt(prompt_dict["current_prompt"],
                                                                                      open_bracket_idx,
                                                                                      closed_bracket_idx)
                    if adapter_name is not None and adapter_name not in loras.keys():
                        loras[adapter_name] = adapter_scale
                    lora_string = prompt_dict["current_prompt"][open_bracket_idx:closed_bracket_idx + 1]
                    replaced = prompt_dict["current_prompt"].replace(lora_string, "")
                    prompt_dict["current_prompt"] = replaced
                    break
            if idx >= len(prompt_dict["current_prompt"]) - 1:
                eos = True
                break
        if eos:
            break

    return prompt_dict["current_prompt"], loras


def create_lora_params_from_prompt(prompt, open_bracket_idx, closed_bracket_idx):
    # add 1 to start from the first letter after the open bracket
    lora_string = prompt[open_bracket_idx+1:closed_bracket_idx]
    if len(lora_string) == 0:
        return None, None  # adapter_name, adapter_scale

    # check if colon separator exists for lora scale value
    if ":" in lora_string:
        adapter_name, adapter_scale = lora_string.split(":")
    else:
        adapter_name = lora_string
        adapter_scale = 1.0

    return adapter_name, adapter_scale


class WanVideoGenerator(VideoGenerator):

    JOB_ID = ""
    STEPS_MAX = 0
    STEPS_CURRENT = 0
    NUM_FRAMES_ITER = 0
    NUM_FRAMES_MAX = 0

    def __init__(self, is_server=False, local_files_only=True, model="wan-480"):
        '''

        :param is_server: determines the output name in redresser_output_file_path
        :param local_files_only:
        '''
        super().__init__(is_server, local_files_only, model)
        self.is_server = is_server
        self.model = model
        self.dtype = torch.bfloat16
        # initialize default settings first
        if model == "wan-480":
            self.settings = WanSettings()
            # self.load_quanto_pipe()
            self.pipe = None
            self.load_pipe()

    def load_pipe(self):
        model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

        quant_config = BitsAndBytesConfig(load_in_8bit=True)

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
                model_id, subfolder="vae", torch_dtype=torch.float32)


        # original
        # if os.path.exists("D:\\huggingface\\models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers\\snapshots\\ba97433dcc621976ffdde384974f890f64190b18"):
        #     with tqdm(desc="Loading original text_encoder from Wan-AI--Wan2.1-I2V-14B-480P-Diffusers"):
        #         text_encoder = UMT5EncoderModel.from_pretrained(
        #             "D:\\huggingface\\models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers\\snapshots\\ba97433dcc621976ffdde384974f890f64190b18",
        #             subfolder="text_encoder",
        #             torch_dtype=torch.bfloat16
        #         )
        # else:
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
            text_encoder.to(torch.bfloat16)

        with tqdm(desc="Loading transformer"):
            # DO NOT USE QUANTIZED 1.3B, OUTPUT IS JUST NOISE
            # transformer = WanVACETransformer3DModel.from_single_file(
            #     "https://huggingface.co/calcuis/wan-gguf/blob/main/wan2.1-v4-vace-1.3b-q4_0.gguf",
            #     # quantization_config=quant_config,
            #     quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            #     torch_dtype=torch.bfloat16)

            # transformer = WanVACETransformer3DModel.from_pretrained(
            #     "Wan-AI/Wan2.1-VACE-1.3B-diffusers", subfolder="transformer",
            #     torch_dtype=torch.bfloat16)

            transformer = WanVACETransformer3DModel.from_single_file(
                "https://huggingface.co/QuantStack/Wan2.1_14B_VACE-GGUF/blob/main/Wan2.1_14B_VACE-Q4_0.gguf",
                # quantization_config=quant_config,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16)

            # transformer = WanTransformer3DModel.from_single_file(
            #     "https://huggingface.co/QuantStack/Phantom_Wan_14B_FusionX-GGUF/blob/main/Phantom_Wan_14B_FusionX-Q4_0.gguf",
            #     # quantization_config=quant_config,
            #     quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            #     torch_dtype=torch.bfloat16)
            # transformer = WanTransformer3DModel.from_single_file(
            #     "https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/wan2.1-i2v-14b-480p-Q4_0.gguf",
            #     # quantization_config=quant_config,
            #     quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            #     torch_dtype=torch.bfloat16)

        # self.pipe = WanImageToVideoPipeline.from_pretrained(
        #     model_id,
        #     text_encoder=text_encoder, tokenizer=tokenizer,
        #     transformer=transformer, vae=vae, image_encoder=image_encoder,
        #     torch_dtype=torch.bfloat16
        # )
        #
        # flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
        # self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=flow_shift)

        self.pipe = WanVACEPipeline.from_pretrained(
            model_id,
            text_encoder=text_encoder, tokenizer=tokenizer,
            transformer=transformer, vae=vae,
            torch_dtype=torch.bfloat16
        )

        flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=flow_shift)

        # self.pipe.enable_model_cpu_offload()

        # group-offloading
        onload_device = torch.device("cuda")
        offload_device = torch.device("cpu")
        apply_group_offloading(text_encoder,
                               onload_device=onload_device,
                               offload_device=offload_device,
                               offload_type="block_level",
                               num_blocks_per_group=4
                               )
        transformer.enable_group_offload(
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type="leaf_level",
            use_stream=True
        )
        self.pipe.to("cuda")

    @staticmethod
    def prepare_video_and_mask(frame_inserts: dict, height: int, width: int,
                               num_frames: int):

        first_img = PIL.Image.new("RGB", (width, height), (128, 128, 128))
        last_img = PIL.Image.new("RGB", (width, height), (128, 128, 128))

        frames = [first_img]
        # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
        # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
        # match the original code.
        frames.extend([PIL.Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 2))
        frames.append(last_img)

        # insert the frames
        for frame_index, frame in frame_inserts.items():
            frames[frame_index] = frame

        mask_black = PIL.Image.new("L", (width, height), 0)
        mask_white = PIL.Image.new("L", (width, height), 255)
        mask = [mask_black, *[mask_white] * (num_frames - 2), mask_black]
        return frames, mask

    def run(self, video=None, mask=None, references=None, frame_index=0, video_fps=16):
        # this is used for saving the output
        image_path = self.settings.options.get("image")

        seed = int(self.settings.options.get("seed", -1))
        if seed is None or seed < 0:
            seed = int(time.time())

        print("seed", seed)
        generator = torch.Generator().manual_seed(seed)

        print("Running with setting:", self.settings.options)

        # filter and extract loras from prompt and apply them to the pipe
        prompt = self.settings.options["prompt"]
        prompt, loaded_loras = apply_flux_loras_with_prompt(self.pipe, prompt)

        negative_prompt = "anime, cartoon, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        args = {"prompt": prompt, "negative_prompt": negative_prompt,
                "num_frames": self.settings.options["num_frames"],
                "guidance_scale": self.settings.options["guidance_scale"],
                "num_inference_steps": self.settings.options["num_inference_steps"],
                }

        if references:
            args["reference_images"] = references

        if video is None or mask is None:
            args["conditioning_scale"] = 0.6
            height = self.settings.options["height"]
            width = self.settings.options["width"]
        else:
            args["video"] = video
            if mask[0] is not None:
                args["mask"] = mask
            height = video[0].height
            width = video[0].width

        args["height"] = height
        args["width"] = width

        if generator is not None:
            args["generator"] = generator

        flow_shift = self.settings.options.get("flow_shift", 2.0)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=flow_shift)

        callback_on_step_end = self.settings.options.get("callback_on_step_end", None)
        if callback_on_step_end:
            args["callback_on_step_end"] = callback_on_step_end

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

            def dict_to_filename(data: dict, separator="-", kv_separator="_"):
                return separator.join(f"{k}{kv_separator}{v}" for k, v in data.items())
            input_file_name = f"{redresser_dir}/{seed}_{os.path.basename(image_path)}_{self.settings.options['guidance_scale']}_{self.settings.options['num_inference_steps']}_{width}x{height}_{frame_index}_{self.settings.options['num_frames']}_{dict_to_filename(loaded_loras)}.mp4"

        if not os.path.exists(redresser_dir):
            os.mkdir(redresser_dir)
        output_file_name = f"{input_file_name}-opt.mp4"

        # # skip first frame
        # try:
        #     _legacy_export_to_video(video.frames[0][1:], input_file_name, fps=video_fps)
        #     # run_ffmpeg_optical_flow(input_video=input_file_name, output_video=output_file_name)
        #
        # except Exception as e:
        #     print(e)
        #     result_dir = f"{redresser_dir}/{seed}"
        #     if not os.path.exists(result_dir):
        #         os.mkdir(result_dir)
        #     for f_idx, frame in enumerate(video.frames[0]):
        #
        #         frame.save(f"{result_dir}/{str(f_idx).zfill(5)}.png")

        return video.frames[0], input_file_name

    def load_and_prepare_image(self, img_path):

        if isinstance(img_path, np.ndarray):
            image = Image.fromarray(img_path)
        else:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        # max_area = 480 * 832
        max_area = self.settings.options.get("max_area", 832*480)
        aspect_ratio = image.height / image.width
        mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))
        return image

    def process_fill_outputs(self, orig_imgs, seg_imgs, outputs, yolo_results):
        final_pil_images = []
        final_cv2_images = []
        segment_id = self.settings.options["SEGMENT_ID"]
        for image_idx, image in enumerate(outputs):

            np_image = None

            # paste each original face back one by one; skip if segmenting face

            for (face_index, face_data) in yolo_results[image_idx].items():
                # no keys = nothing extracted
                if segment_id != RedresserSettings.SEGMENT_FACE and self.settings.options["keep_face"]:
                    if np_image is None:
                        np_image = np.array(image)

                    if 'aligned_cropped_image' not in face_data.keys():
                        continue
                    swapped_image = face_data[
                        "aligned_cropped_image"]  # unsharp_mask(extracted_data["aligned_cropped_image"], amount=.5)
                    seg_mask = face_data["seg_mask"]
                    aligned_cropped_params = face_data["aligned_cropped_params"]
                    np_image = paste_swapped_image(
                        dst_image=np_image,
                        swapped_image=swapped_image,
                        seg_mask=seg_mask,
                        aligned_cropped_params=aligned_cropped_params,
                        seamless_clone=False,
                        blur_mask=True,
                        resize=False
                    )
                else:
                    if np_image is None:
                        np_image = np.array(orig_imgs[image_idx])
                    swapped_image = np.array(image)  # unsharp_mask(extracted_data["aligned_cropped_image"], amount=.5)
                    h, w = swapped_image.shape[:2]
                    seg_mask = np.array(seg_imgs[image_idx])
                    # seg_mask = cv2.blur(seg_mask, (3, 3))
                    sigma = 5
                    kernel_size = 20
                    if kernel_size % 2 == 0:
                        kernel_size += 1  # must be odd for cv2

                    seg_mask = cv2.GaussianBlur(seg_mask, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)

                    # seg_mask = cv2.GaussianBlur(seg_mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
                    seg_mask = np.clip(seg_mask, 0, 255)

                    if seg_mask.shape[-1] > 1:
                        seg_mask = seg_mask[..., 0]
                    if len(seg_mask.shape) == 2:
                        seg_mask = np.expand_dims(seg_mask, axis=-1)

                    seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_CUBIC)
                    np_image = cv2.resize(np_image, (w, h), interpolation=cv2.INTER_CUBIC)

                    np_image = paste_image_with_mask(np_image, swapped_image, seg_mask)

                    # center = (np_image.shape[1]//2, np_image.shape[0]//2)
                    # np_image = cv2.seamlessClone(swapped_image, np_image, seg_mask,
                    #                                  center, cv2.NORMAL_CLONE)

            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
            final_cv2_images.append(np_image)
            #
            image = Image.fromarray(cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)).convert('RGB')
            final_pil_images.append(image)

        return final_pil_images


class VideoAnnotator(ImageProcessor):
    def __init__(self):
        super().__init__()

    def prepare_video(self, video_path, settings, start_frame=0, frames=81, frame_skip=0, segment_id=RedresserSettings.SEGMENT_FACE):
        file_name = os.path.basename(video_path)
        ext = file_name.split(".")[-1]
        file_name_wo_ext = file_name.replace(f".{ext}", "")
        video_dir = video_path.replace(file_name, "")
        seg_image_dir = os.path.join(video_dir, file_name_wo_ext)

        os.makedirs(seg_image_dir, exist_ok=True)

        print(f"reading video from {video_path}")
        cap = cv2.VideoCapture(video_path)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_index = start_frame
        current_index = 0

        orig_imgs = []
        seg_imgs = []

        height = width = None

        skip_counter = frame_skip
        while True:
            ret, frame = cap.read()

            if ret:
                # skip frames
                if skip_counter < frame_skip:
                    skip_counter += 1
                    frame_index += 1
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # set width and height
                if height is None or width is None:
                    image = Image.fromarray(frame)
                    # max_area = 480 * 832
                    max_area = settings.options.get("max_area", 832 * 480)
                    aspect_ratio = image.height / image.width
                    mod_value = settings.options["image_mod_value"]
                    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
                    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

                # face segmentation
                yolo_results = yolo8_extract_faces(
                    face_extractor=self.face_extract_model, face_seg_model=self.face_mask_model, max_faces=10,
                    conf_threshold=0.45, landmarks_padding_ratio=settings.options.get("face_mask_scale", 1.1),
                    inputs=[frame],
                    include_orig_img=False,
                    # resizes outputs
                    face_swapper_input_size=(256, 256)  # set to xseg output size so xseg doesn't resize again
                )
                try:
                    face_masks_combined = yolo_results[0][0]['seg_mask_combined']
                except KeyError:
                    face_masks_combined = np.zeros_like(frame)

                # redresser segmentation
                orig_img, seg_img = self.process_image_for_video_segmentation(
                    PIL.Image.fromarray(frame),
                    seg_dir=seg_image_dir,
                    frame_index=frame_index,
                    face_masks_combined=face_masks_combined,
                    segment_id=segment_id
                )

                # add to batch
                orig_imgs.append(orig_img.copy())
                seg_imgs.append(None if seg_img is None else seg_img.copy())
                print(f"processed frame {frame_index}")
                frame_index += 1

                # reset skip counter
                if skip_counter >= frame_skip:
                    skip_counter = 0

            if not ret or frames == len(orig_imgs):
                break

        cap.release()
        print("processed frames", len(orig_imgs))
        # modify original images from masks (pose does npt use masks)
        if segment_id != RedresserSettings.POSE_FULL:
            for img_index in range(len(orig_imgs)):
                new_frame = PIL.Image.new("RGB", (width, height), (128, 128, 128))
                mask_inverse = seg_imgs[img_index].point(lambda p: 255 - p).convert("L")
                new_frame.paste(orig_imgs[img_index], mask=mask_inverse)
                orig_imgs[img_index] = new_frame

        return orig_imgs, seg_imgs, not ret

    def process_image_for_video_segmentation(self, image, seg_dir, frame_index, face_masks_combined, segment_id):
        '''
        seg_mask_combined(ndarray uint8): the combined segmented face masks for this image
        '''
        seg_face_file = os.path.join(seg_dir, f"{frame_index}-seg-face.png")
        seg_person_file = os.path.join(seg_dir, f"{frame_index}-seg-person.png")
        seg_fashion_file = os.path.join(seg_dir, f"{frame_index}-seg-fashion.png")
        seg_hands_file = os.path.join(seg_dir, f"{frame_index}-seg-hands.png")
        pose_full_file = os.path.join(seg_dir, f"{frame_index}-pose-full.png")
        seg_combined_file = os.path.join(seg_dir, f"{frame_index}-seg-combined.png")
        # image should be in RGB

        if segment_id == RedresserSettings.SEGMENT_FACE:
            if os.path.exists(seg_face_file):
                seg_img = Image.open(seg_face_file)
                seg_img = seg_img.resize((image.width, image.height))
            else:
                # process and segment the face
                _, seg_img = yolo_segment_image(self.head_seg_model, np.array(image), return_original_image=False)[0]

                seg_img = expand_mask(face_masks_combined, (5, 5))
                seg_img = Image.fromarray(seg_img)
                seg_img.save(seg_face_file)
                seg_img = seg_img.resize((image.width, image.height))
            seg_img.save(seg_combined_file)
            return image, seg_img
        elif segment_id == RedresserSettings.SEGMENT_ALL:
            # use all white as the mask
            seg_img = Image.new("RGB", (image.width, image.height), (255, 255, 255))
            seg_img.resize((image.width, image.height))

        elif segment_id in (RedresserSettings.SEGMENT_PERSON, RedresserSettings.SEGMENT_FASHION, RedresserSettings.SEGMENT_BG):
            # process and segment the image (person/fashion)
            if segment_id in (RedresserSettings.SEGMENT_PERSON, RedresserSettings.SEGMENT_BG):
                seg_file = seg_person_file
            elif segment_id == RedresserSettings.SEGMENT_FASHION:
                seg_file = seg_fashion_file

            if os.path.exists(seg_file):
                seg_img = Image.open(seg_file)
                seg_img = seg_img.resize((image.width, image.height))
            else:
                _, seg_img = yolo_segment_image(self.f_seg_model, np.array(image), return_original_image=False)[0]
                seg_img = np.array(seg_img, dtype=np.uint8)
                if segment_id == RedresserSettings.SEGMENT_FASHION:
                    m = (seg_img.shape[0] + seg_img.shape[1]) // 32
                    if m % 2 == 0:
                        m = m + 1
                    seg_img = expand_mask(seg_img, (m, m))

                # test here to use noise on original where mask is
                # _orig_img[seg_img == 255] = np.random.randint(0, 256)
                seg_img = Image.fromarray(seg_img)
                seg_img.save(seg_file)

            if segment_id == RedresserSettings.SEGMENT_BG:
                if seg_img.mode != "L":
                    seg_img = seg_img.convert("L")
                seg_img = ImageOps.invert(seg_img)
                seg_img.convert("RGB")
        elif segment_id == RedresserSettings.POSE_FULL:

            if os.path.exists(pose_full_file):
                image = Image.open(pose_full_file)

                image = image.resize((image.width, image.height))
            else:
                p = Processor("openpose")  # openpose_face
                image = p(image)
                image.save(pose_full_file)

            return image, None

        else:
            raise ValueError("invalid segment id")

        # process and segment the image (hands)
        if self.settings.options["keep_hands"]:
            if os.path.exists(seg_hands_file):
                seg_img_hands = Image.open(seg_hands_file)
                seg_img_hands = seg_img_hands.resize((image.width, image.height))
            else:
                _, seg_img_hands = yolo_segment_image(self.hand_seg_model, np.array(image), return_original_image=False)[0]
                m = (seg_img_hands.shape[0] + seg_img_hands.shape[1]) // 512
                if m % 2 == 0:
                    m = m + 1
                seg_img_hands = expand_mask(seg_img_hands, (m, m))
                seg_img_hands = Image.fromarray(seg_img_hands)
                seg_img_hands.save(seg_hands_file)

        # add face masks
        if face_masks_combined is not None:
            # resize convert to numpy
            seg_img = seg_img.resize((image.width, image.height))
            seg_img = np.array(seg_img, dtype=np.uint8)

            if self.settings.options["keep_face"]:  # removes faces from seg; used to keep faces intact or for face restore
                # convert to numpy
                face_mask = face_masks_combined#cv2.resize(face_masks_combined, (image_resize_params.new_w, image_resize_params.new_h), interpolation=cv2.INTER_CUBIC)
                m = (face_mask.shape[0] + face_mask.shape[1]) // 512
                if m % 2 == 0:
                    m = m + 1
                face_mask = expand_mask(face_mask, (m, m))
                # seg_img = np.array(seg_img, dtype=np.uint8)
                seg_img[face_mask > 25] = 0  # set all pixels to black if face_mask(white fill) is above noise level

                # if seg_path is not None and SAVE_SEG_IMAGES:
                #     Image.fromarray(face_mask).save(seg_path + "-face_mask.png")

            if self.settings.options["keep_hands"]:  # removes hands from seg; used to keep hands intact
                # convert to numpy
                seg_img_hands = seg_img_hands.resize((image.width, image.height))
                seg_img_hands = np.array(seg_img_hands, dtype=np.uint8)

                seg_img[seg_img_hands > 127] = 0

            # convert back to PIL
            seg_img = Image.fromarray(seg_img).convert('RGB')

        seg_img.save(seg_combined_file)

        return image, seg_img


class WanSettings(RedresserSettings):

    default_options = {
        "prompt": "",
        "image": "jjk.png",

        "max_area": 920 * 920,  # HD size
        # "max_area": 832 * 480,  # HD size
        # "max_area": 640 * 368,  # preview/sample size
        "width": 704,
        "height": 576,
        # "width": 480,
        # "height": 832,
        # "max_area": 640 * 368,
        # "width": 368,
        # "height": 640,
        "guidance_scale": 1.0,
        "num_inference_steps": 4,
        "num_frames": 17,
        # "seed": 1752001920,
    }

    def __init__(self):
        super().__init__()
        self.previous_options = WanSettings.default_options.copy()


if __name__ == "__main__":
    wan = WanVideoGenerator(is_server=False,)
    wan.settings = WanSettings()
    wan.settings.options = WanSettings.default_options.copy()
    wan.settings.options["image_mod_value"] = wan.pipe.vae_scale_factor_spatial * \
                                              wan.pipe.transformer.config.patch_size[1]  # latent dims
    # control video options
    stream_video = True
    runs = 8
    start_frame = 0
    frame_skip = 0
    wan.settings.options["SEGMENT_ID"] = RedresserSettings.SEGMENT_FASHION
    wan.settings.options["keep_face"] = False
    wan.settings.options["keep_hands"] = False


    # wan.settings.options["prompt"] = \
    #     "<wan-fusionx:0.5><wan-cowgirl:0.5>a completely naked 13-year-old girl is on top of another girl,getting fucked in C0WG1RL position on the ground of a dark dirty alley, cock is visibly sliding in and out of her pussy"
    # wan.settings.options["prompt"] = \
    #     "<wan-fusionx>14-year-old girl completely naked, surrounded in glowing and glimmering fairy dust, dancing in a magic forest with wild fantastical beasts wondering in the background, the sky is a celestial aurora"
    # wan.settings.options["prompt"] = \
    #     "<wan-fusionx:1><wan-cowgirl:0.5>14-year-old girl schoolgirl outfit in cowgirl position on expensive sofa, hands tied up behind back, legs raised up, she spreads her legs apart, revealing her vagina, then spreads her vagina labia"
    # wan.settings.options["prompt"] = \
    #     "<wan-fusionx:1>shot of a girl from behind bent over showing her butt to the camera while looking at you,she spreads her pussy"
    # wan.settings.options["prompt"] = \
    #     "<wan-fusionx>14-year-old girl completely naked, surrounded in celestial aurora, dancing"
    wan.settings.options["prompt"] = \
        "<wan-fusionx:1>14-year-old girl completely naked in cowgirl position with a dildo inside her pussy on office table, hands tied up behind back, legs raised up, she spreads her legs apart, revealing her vagina, then spreads her vagina labia"
    wan.settings.options["prompt"] = \
        "<wan-fusionx:1><wan-fingering:0.1>12-year-old skinny korean girls sitting on another girls panting lustfully at you, in Times Square completely naked with legs raised up and apread apart fingering her vagina, "
    # wan.settings.options["prompt"] = \
    #     "<wan-fusionx:1>14-year-old girls completely naked kissing each other passionately"
    # wan.settings.options["prompt"] = \
    #     "<wan-fusionx:1>14-year-old girls completely naked with eyes rolled(looking) upwards and mouths open wide and tongue sticking out all the way with a cum shot on her face"
    # wan.settings.options["prompt"] = \
    #     "<wan-fusionx:1>A hand is in the middle of the frame, then swirling ball of fire begins to erupt from the palm, growing larger until it detonates into a giant fiery explosion"

    reference_images = [
        # Image.open("jjk_3.png").convert("RGB"),
        # Image.open("jjk_1.png").convert("RGB"),

        Image.open("cp_reference/cp_2.png").convert("RGB"),
        Image.open("cp_reference/cp_3.png").convert("RGB"),
        Image.open("cp_reference/cp_5.png").convert("RGB"),
        Image.open("cp_reference/cp_6.png").convert("RGB"),
        # Image.open("C:/Users/teckt/Documents/references/hand.jpg").convert("RGB"),
        # Image.open("C:/Users/teckt/Documents/references/fireball.jpg").convert("RGB"),
        # Image.open("C:/Users/teckt/Documents/references/choking_0.jpg").convert("RGB"),

    ]
    # reference_images = None


    # for i in range(runs):
    #     if i > 0:
    #         wan.settings.options["seed"] = -1
    #         # wan.settings.options["prompt"] = \
    #         #     "<wan-fusionx>14-year-old girl corpse laying on basement floor, large bloody gash in neck and stomach"
    #
    #     wan.run(references=reference_images, video_fps=16)
    #
    # exit()

    # video_path = "C:/Users/teckt/PycharmProjects/iae_dfstudio/yolomodel/fox0.ts"
    # video_path = "C:/Users/teckt/PycharmProjects/iae_dfstudio/yolomodel/worada_0.mp4"
    # video_path = "C:/Users/teckt/PycharmProjects/iae_dfstudio/yolomodel/boeun7.mp4"
    # video_path = "C:/Users/teckt/Documents/webcam/loolashow-20250704.152919.ts"
    video_path = "video/may_1.mp4"
    # video_path = "video/sarang_3.mp4"

    ########
    ## V2V
    ########

    cap = cv2.VideoCapture(video_path)
    video_fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    cap.release()

    output_fps = int(video_fps/(frame_skip+1))

    print(f"video_fps: {video_fps}, output_fps: {output_fps}")

    image_proc = VideoAnnotator()
    if wan.settings.options["SEGMENT_ID"] != RedresserSettings.POSE_FULL:
        image_proc.set_seg_models(wan.settings)
    wan.settings.options["prompt"] = \
        "<wan-fusionx:1><wan-cowgirl:0.2>13-year-old korean girl completely naked, dancing"
    all_video_frames = []
    for i in range(runs):
        while True:
            video, mask, reached_end_of_video = image_proc.prepare_video(
                video_path,
                wan.settings, start_frame=start_frame, frames=wan.settings.options["num_frames"], frame_skip=frame_skip,
                segment_id=wan.settings.options["SEGMENT_ID"]
            )
            if len(video) < 5:
                print(f"reached end of video, cannot process unless frames are in multiples of 4 + 1, remainder frames={len(video)}")
                break

            # adjust num_frames if less
            if len(video) < wan.settings.options['num_frames']:
                num_frames_mod = ((len(video)-1) % 4)  # mod after subtracting 1
                if num_frames_mod > 0:
                    video = video[:-num_frames_mod]
                    mask = mask[:-num_frames_mod]
                wan.settings.options["num_frames"] = len(video)
                print(f"adjusted num_frames [{wan.settings.options['num_frames'] }] to {len(video)}")

            if len(all_video_frames) > 0:
                # replace first frame with previous frame
                prev_frame = all_video_frames[-1].copy()*255
                video[0] = Image.fromarray(prev_frame.astype("uint8"))
                if wan.settings.options["SEGMENT_ID"] != RedresserSettings.POSE_FULL:
                    mask[0] = PIL.Image.new("RGB", (video[0].width, video[0].height), 0)
            print("start_frame=", start_frame, "len(video)=", len(video), "wan.settings.options['num_frames']=", wan.settings.options['num_frames'])
            output, _ = wan.run(video, mask, references=reference_images, frame_index=start_frame, video_fps=output_fps)

            if not stream_video:
                break

            # discard first frame if continuing (the first frame is always darker)
            if len(all_video_frames) > 0:
                output = output[1:]

            for _f in output:
                all_video_frames.append(_f)

            # export full video
            _legacy_export_to_video(all_video_frames, video_path+"_wan_full.mp4", fps=output_fps)
            # run_ffmpeg_optical_flow(input_video=video_path+"_wan_full.mp4", output_video=video_path+"_wan_full_opt.mp4", fps=int(video_fps/(frame_skip+1) * 2))

            if reached_end_of_video:
                print(f"reached end of video, remainder frames={len(video)}")
                break

            # set the next start frame ( minus one because we're going to use the last frame as the first frame)
            if frame_skip > 0:
                start_frame = start_frame + len(video) * (frame_skip+1) - (frame_skip+1)  # add one to frame skip (skipping 3 means every 4th)
            else:
                start_frame = start_frame + len(video) - 1

    ########
    ## FROM VIDEO FRAME
    ########
    wan.settings.options["prompt"] = \
        "<wan-fusionx:1>cinematic camera of man using both hands to choke the girl, she has a painful expression, he pushes her to the ground"
    video_path = "video_results/1752200580_jjk.png_1.0_6_688x560_0_81_wan-fusionx_1.0.mp4"
    wan.settings.options["prompt"] = \
        "<wan-fusionx:1><wan-fingering:0.25>13-year-old asian girl completely naked sits down, raises her legs and spreads it apart, she starts fingering her vagina"
    video_path = "video/water_1.mp4_wan_full.mp4"

    cap = cv2.VideoCapture(video_path)
    frame_idx = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    cv2.imwrite("C:/Users/teckt/Documents/references/video_frame.png", frame)
    base_frame = wan.load_and_prepare_image("C:/Users/teckt/Documents/references/video_frame.png")

    frame_inserts = {}
    frame_inserts[0] = base_frame
    video, mask = WanVideoGenerator.prepare_video_and_mask(
        frame_inserts=frame_inserts,
        height=base_frame.height, width=base_frame.width, num_frames=wan.settings.options["num_frames"]
    )

    for i in range(runs):
        if i > 0:
            wan.settings.options["seed"] = -1
        _, output_file_name = wan.run(video, mask, references=reference_images, video_fps=16)
        # export full video
        concatenate_videos(video_path1=video_path, video_path2=output_file_name, output_path=output_file_name+"_LF2V.mp4")

        # _legacy_export_to_video(output[1:], video_path+"_wan_full.mp4", fps=16)

    # ########
    # ## I2V
    # ########
    # base_frame = wan.load_and_prepare_image("C:/Users/teckt/Documents/references/hand.jpg")
    # frame_inserts = {}
    # frame_inserts[0] = base_frame
    # video, mask = WanVideoGenerator.prepare_video_and_mask(
    #     frame_inserts=frame_inserts,
    #     height=base_frame.height, width=base_frame.width, num_frames=wan.settings.options["num_frames"]
    # )
    #
    # for i in range(runs):
    #     if i > 0:
    #         wan.settings.options["seed"] = -1
    #     wan.run(video, mask, references=reference_images, video_fps=16)

