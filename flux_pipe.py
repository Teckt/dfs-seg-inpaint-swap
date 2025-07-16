import os
import sys
import time

import huggingface_hub
from PIL import Image
import numpy as np
import cv2
import torch
from diffusers import FluxFillPipeline, AutoencoderKL, FluxTransformer2DModel, FluxPipeline, FluxKontextPipeline, \
    GGUFQuantizationConfig
from huggingface_hub import hf_hub_download
# from diffusers.utils import load_image
from tqdm import tqdm
from transformers import T5EncoderModel, CLIPTextModel
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as BitsAndBytesConfig
from CONSTANTS import *
from optimum.quanto import freeze, qfloat8, quantize

from fire_functions import FirestoreFunctions


# from optimum_quanto.optimum.quanto import QuantizedTransformersModel


# from yolomodel.redresser.flux_controlnet_inpainting.transformer_flux import FluxTransformer2DModel as FluxTransformer2DModel_alimama
# from yolomodel.redresser.flux_controlnet_inpainting.controlnet_flux import FluxControlNetModel
# from yolomodel.redresser.flux_controlnet_inpainting.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline


class MyFluxPipe:

    def __init__(self, fill=True, use_hyper=False, use_kontext=False):
        self.fire_functions = FirestoreFunctions()

        self.dtype = torch.bfloat16
        self.lora_path = LORA_PATH
        self.is_fill = fill
        self.use_kontext = use_kontext
        self.use_hyper = use_hyper
        self.flux_model_name = self.get_flux_model_name()

        # load VAE
        # with tqdm(range(1), "Loading flux_vae"):
        #     self.flux_vae = AutoencoderKL.from_pretrained(
        #         self.flux_model_name, subfolder="vae",
        #         # "C:/Users/teckt/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/ae.safetensors"
        #         # "C:\\Users\\teckt\\.cache\\huggingface\\hub\\models--Kijai--flux-fp8\\snapshots\\e77f550e3fe8be226884d5944a40abdbe4735ff5\\flux1-dev-fp8.safetensors",
        #         torch_dtype=self.dtype, local_files_only=True)
        #     # self.flux_vae.to(self.dtype)

        self.fused_turbo = False


        self.pipes = {"t2i":None, "fill":None, "kontext":None}

        # load tokenizer
        with tqdm(range(1), "Loading CLIPTextModel") as progress_bar:
            self.clip_L_text_encoder = CLIPTextModel.from_pretrained(
                TEXT_ENCODER_CLIP_L_PATH,
                torch_dtype=self.dtype,
                local_files_only=USE_LOCAL_FILES)
            progress_bar.update()

        # check if model name is Anyfusion and exists locally without the repo id
        text_encoder_args = {
            "torch_dtype": self.dtype, "local_files_only": USE_LOCAL_FILES
        }
        if os.path.exists(os.path.join("t5-encoder-fp16", "model.safetensors")):
            text_encoder_id = "t5-encoder-fp16"
        else:
            text_encoder_id = FLUX_PATH
            text_encoder_args["subfolder"] = "text_encoder_2"

        with tqdm(total=1, desc="loading text_encoder_2") as progress_bar:
            self.text_encoder_2 = T5EncoderModel.from_pretrained(
                text_encoder_id, **text_encoder_args)
            progress_bar.update()

        if self.use_kontext:
            pretrained_model_name_or_path = "black-forest-labs/FLUX.1-Kontext-dev"
        else:
            pretrained_model_name_or_path = FLUX_FILL_PATH if fill else FLUX_PATH
        pipeline_args = {
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "transformer": None,
            "text_encoder": self.clip_L_text_encoder,
            "text_encoder_2": None,
            "torch_dtype": self.dtype,
            "local_files_only": USE_LOCAL_FILES
        }

        if self.use_kontext:
            transformer = FluxTransformer2DModel.from_single_file(
                "https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF/blob/main/flux1-kontext-dev-Q4_0.gguf",
                # quantization_config=quant_config,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16)
        else:
            transformer = self.load_transformer()


        print("loading pipeline")

        if fill:
            if self.use_kontext:
                self.pipes["kontext"] = FluxKontextPipeline.from_pretrained(**pipeline_args)
                self.pipe = self.pipes["kontext"]
            else:
                self.pipes["fill"] = FluxFillPipeline.from_pretrained(**pipeline_args)
                self.pipe = self.pipes["fill"]
        else:
            self.pipes["t2i"] = FluxPipeline.from_pretrained(**pipeline_args)
            self.pipe = self.pipes["t2i"]

        print("adding transformer")
        self.pipe.transformer = transformer
        print("adding t5 encoder")
        self.pipe.text_encoder_2 = self.text_encoder_2


        ### DEBUG ONLY ###
        # # fuse lora before quantizing
        # if FUSE_HYPER_LORA:
        #     self.fuse_hyper_lora()
        # # quantize if original flux models
        # if not USE_BNB and USE_OPTIMUM_QUANTO and ((self.flux_model_name == FLUX_PATH and not self.is_fill) or (self.flux_model_name == FLUX_FILL_PATH and self.is_fill)):
        #     self.quanto_quantize()
        #
        # if SAVE_MODEL and not os.path.exists(
        #         f"{SAVE_MODEL_PATH}/diffusion_pytorch_model.safetensors"):
        #     # save the model here
        #     with tqdm(range(1), "Saving transformer"):
        #         print(f"saving transformer to {SAVE_MODEL_PATH}")
        #         transformer.save_pretrained(SAVE_MODEL_PATH, max_shard_size=SHARD_SIZE)
        ### DEBUG ONLY ###


        self.pipe.vae.enable_slicing()
        # self.pipe.vae.enable_tiling()

        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(device)
            total_memory = torch.cuda.get_device_properties(device).total_memory
            VRAM = total_memory / (1024 ** 3)
            print(f"GPU Name: {gpu_name}")
            print(f"Total Memory: {total_memory / (1024 ** 3):.2f} GB")  # Convert bytes to GB
        else:
            VRAM = 0
            print("CUDA is not available.")


        ### DEBUG ONLY ###
        # if USE_OPTIMUM_QUANTO:
        #     if USE_CPU_OFFLOAD:
        #         self.pipe.enable_model_cpu_offload()
        #     elif USE_SEQUENTIAL_CPU_OFFLOAD:
        #         self.pipe.enable_sequential_cpu_offload()
        #     else:
        #         self.pipe.to("cuda")
        # # move to cuda if original flux models only
        # elif USE_BNB and ((self.flux_model_name == FLUX_PATH and not self.is_fill) or (self.flux_model_name == FLUX_FILL_PATH and self.is_fill)):
        #     self.pipe.to("cuda")
        #     # self.pipe.enable_model_cpu_offload()
        # else:
        #     if USE_CPU_OFFLOAD:
        #         self.pipe.enable_model_cpu_offload()
        #     elif USE_SEQUENTIAL_CPU_OFFLOAD:
        #         self.pipe.enable_sequential_cpu_offload()
        #     else:
        #         self.pipe.to("cuda")
        ### DEBUG ONLY ###

        if USE_CPU_OFFLOAD:
            self.pipe.enable_model_cpu_offload()
        elif USE_SEQUENTIAL_CPU_OFFLOAD:
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.to("cuda")

    def get_flux_model_name(self):
        """
        gets the model path for huggingface flux model or custom flux/hyper models
        :return:
        """
        if self.use_kontext:
            if self.use_hyper:
                return FLUX_FILL_HYPER_CUSTOM_PATH if USE_CUSTOM_FLUX_FILL else FLUX_FILL_PATH
            else:
                return FLUX_FILL_CUSTOM_PATH if USE_CUSTOM_FLUX_FILL else FLUX_FILL_PATH

        if self.is_fill:
            if self.use_hyper:
                return FLUX_FILL_HYPER_CUSTOM_PATH if USE_CUSTOM_FLUX_FILL else FLUX_FILL_PATH
            else:
                return FLUX_FILL_CUSTOM_PATH if USE_CUSTOM_FLUX_FILL else FLUX_FILL_PATH
        else:
            if self.use_hyper:
                return FLUX_HYPER_CUSTOM_PATH if USE_CUSTOM_FLUX else FLUX_PATH
            else:
                return FLUX_CUSTOM_PATH if USE_CUSTOM_FLUX else FLUX_PATH

    def switch_pipeline(self, pipe):
        assert pipe == "fill" or pipe == "t2i"
        if pipe == "fill":
            if self.is_fill:
                print("Already is fill")
                return
            else:
                # switch transformers and pipe
                self.switch_pipe(fill=True)
        else:
            if not self.is_fill:
                print("Already is not fill")
                return
            else:
                # switch transformers and pipe
                self.switch_pipe(fill=False)

    def switch_pipe(self, fill):

        self.is_fill = fill

        self.flux_model_name = self.get_flux_model_name()

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            current_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert to GB
            max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(f"GPU memory allocated: {current_memory:.2f}/{max_memory:.2f} GB")
        else:
            print("CUDA is not available. Running on CPU.")

        # just set pipe if already loaded empty
        fill_pipe_name = "kontext" if self.use_kontext else "fill"
        if fill:
            if self.pipes[fill_pipe_name] is not None:
                if USE_CPU_OFFLOAD or USE_SEQUENTIAL_CPU_OFFLOAD:
                    self.pipe = self.pipes[fill_pipe_name]
                else:
                    print("moving t2i transformer to cpu")
                    self.pipes["t2i"].transformer.to("cpu")
                    current_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert to GB
                    max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    print(f"GPU memory allocated: {current_memory:.2f}/{max_memory:.2f} GB")
                    print("moving fill transformer to cuda")
                    self.pipes[fill_pipe_name].transformer.to("cuda")
                    current_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert to GB
                    max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    print(f"GPU memory allocated: {current_memory:.2f}/{max_memory:.2f} GB")
                    self.pipe = self.pipes[fill_pipe_name]
                print(f"switched pipe to fill")
                return
        else:
            if self.pipes["t2i"] is not None:
                if USE_CPU_OFFLOAD or USE_SEQUENTIAL_CPU_OFFLOAD:
                    self.pipe = self.pipes["t2i"]
                else:
                    self.pipe.transformer.to("cpu")
                    self.pipe = self.pipes["t2i"]
                    self.pipe.transformer.to("cuda")
                print(f"switched pipe to t2i")
                return

        # load models from current pipe except transformer
        if self.use_kontext:
            pretrained_model_name_or_path = "black-forest-labs/FLUX.1-Kontext-dev"
        else:
            pretrained_model_name_or_path = FLUX_FILL_PATH if fill else FLUX_PATH
        pipeline_args = {
            "pretrained_model_name_or_path": pretrained_model_name_or_path,

            "transformer": None,
            "text_encoder": self.pipe.text_encoder,
            "text_encoder_2": self.pipe.text_encoder_2,
            "vae": self.pipe.vae,
            "torch_dtype": self.dtype,
            "local_files_only": USE_LOCAL_FILES
        }

        ### DEBUG ONLY ###
        # # unload to cpu here
        # if USE_OPTIMUM_QUANTO:
        #     if USE_CPU_OFFLOAD:
        #         # self.pipe.disable_model_cpu_offload()
        #         pass
        #     elif USE_SEQUENTIAL_CPU_OFFLOAD:
        #         # self.pipe.disable_sequential_cpu_offload()
        #         pass
        #     else:
        #         self.pipe.transformer.to("cpu")
        # # move to cpu if original flux models only
        # elif USE_BNB and ((self.flux_model_name == FLUX_PATH and not self.is_fill) or (self.flux_model_name == FLUX_FILL_PATH and self.is_fill)):
        #     self.pipe.transformer.to("cpu")
        #     # self.pipe.enable_model_cpu_offload()
        # else:
        #     if USE_CPU_OFFLOAD:
        #         # self.pipe.disable_model_cpu_offload()
        #         pass
        #     elif USE_SEQUENTIAL_CPU_OFFLOAD:
        #         # self.pipe.disable_sequential_cpu_offload()
        #         pass
        #     else:
        #         self.pipe.transformer.to("cpu")
        ### DEBUG ONLY ###

        if USE_CPU_OFFLOAD or USE_SEQUENTIAL_CPU_OFFLOAD:
            pass
        else:
            self.pipe.transformer.to("cpu")

        transformer = self.load_transformer()

        print("switching pipeline")

        if fill:
            if self.use_kontext:
                self.pipes["kontext"] = FluxKontextPipeline.from_pretrained(**pipeline_args)
                self.pipe = self.pipes["kontext"]
            else:
                self.pipes["fill"] = FluxFillPipeline.from_pretrained(**pipeline_args)
                self.pipe = self.pipes["fill"]
        else:
            self.pipes["t2i"] = FluxPipeline.from_pretrained(**pipeline_args)
            self.pipe = self.pipes["t2i"]

        print("adding transformer")
        self.pipe.transformer = transformer

        ### DEBUG ONLY ###
        # if USE_OPTIMUM_QUANTO:
        #     if USE_CPU_OFFLOAD:
        #         self.pipe.enable_model_cpu_offload()
        #     elif USE_SEQUENTIAL_CPU_OFFLOAD:
        #         self.pipe.enable_sequential_cpu_offload()
        #     else:
        #         self.pipe.transformer.to("cuda")
        # # move to cuda if original flux models only
        # elif USE_BNB and ((self.flux_model_name == FLUX_PATH and not self.is_fill) or (self.flux_model_name == FLUX_FILL_PATH and self.is_fill)):
        #     self.pipe.transformer.to("cuda")
        #     # self.pipe.enable_model_cpu_offload()
        # else:
        #     if USE_CPU_OFFLOAD:
        #         self.pipe.enable_model_cpu_offload()
        #     elif USE_SEQUENTIAL_CPU_OFFLOAD:
        #         self.pipe.enable_sequential_cpu_offload()
        #     else:
        #         self.pipe.transformer.to("cuda")
        ### DEBUG ONLY ###


        if USE_CPU_OFFLOAD:
            self.pipe.enable_model_cpu_offload()
        elif USE_SEQUENTIAL_CPU_OFFLOAD:
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.transformer.to("cuda")
        print(f"switched pipe to {'fill' if fill else 't2i'}")

    # def fuse_hyper_lora(self):
    #     # control_pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    #     # control_pipe.load_lora_weights("black-forest-labs/FLUX.1-Depth-dev-lora", adapter_name="depth")
    #
    #     if self.fused_turbo:
    #         print("turbo already fused")
    #         return
    #     with tqdm(range(4), desc="Fusing hyper lora") as p:
    #         if "turbo" in FUSE_HYPER_LORA_REPO.lower():
    #             self.pipe.load_lora_weights(self.lora_path,
    #                                         weight_name="turbo-a.safetensors",
    #                                         adapter_name="turbo")
    #         else:
    #             self.pipe.load_lora_weights(
    #                 hf_hub_download(FUSE_HYPER_LORA_REPO, FUSE_HYPER_LORA_MODEL_FILE),
    #                 adapter_name="turbo"
    #             )
    #
    #         p.update()
    #         p.desc = "Setting adapter"
    #         self.pipe.set_adapters(["turbo"], adapter_weights=[FUSE_HYPER_ALPHA])
    #
    #         p.update()
    #         p.desc = "Fusing lora"
    #         self.pipe.fuse_lora()
    #         p.update()
    #         p.desc = "Unloading lora"
    #         self.pipe.unload_lora_weights()
    #         p.desc = "Deleting adapter"
    #         # self.pipe.delete_adapters(["hyper-sd"])
    #         self.pipe.delete_adapters(["turbo"])
    #         p.update()
    #     self.fused_turbo = True

    def load_transformer(self):
        transformer_args = {
            "torch_dtype": self.dtype, "local_files_only": USE_LOCAL_FILES
        }

        if self.is_fill and self.use_kontext:
            transformer = FluxTransformer2DModel.from_single_file(
                "https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF/blob/main/flux1-kontext-dev-Q4_0.gguf",
                # quantization_config=quant_config,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16)
            return transformer

        # only use BNB if original flux models
        if USE_BNB and ((self.flux_model_name == FLUX_PATH and not self.is_fill) or (self.flux_model_name == FLUX_FILL_PATH and self.is_fill)):
            quant_config = DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            transformer_args["quantization_config"] = quant_config

        # check if model name is Anyfusion and exists locally without the repo id
        if "Anyfusion/" in self.flux_model_name:
            local_dir = self.flux_model_name.replace("Anyfusion/", "")
            if os.path.exists(os.path.join(local_dir, "diffusion_pytorch_model.safetensors")):
                self.flux_model_name = local_dir

        # load and quantize transformer
        with tqdm(range(1), "Loading and quantizing transformer") as progress_bar:
            # # fp8 kijai
            # transformer = FluxTransformer2DModel.from_single_file(
            #     self.flux_model_name, subfolder="transformer", **transformer_args)
            try:
                transformer = FluxTransformer2DModel.from_pretrained(
                    self.flux_model_name, **transformer_args)
            except OSError:
                transformer = FluxTransformer2DModel.from_pretrained(
                    self.flux_model_name, subfolder="transformer", **transformer_args)
        return transformer

    # def quanto_quantize(self):
    #     # load and quantize transformer
    #     with tqdm(range(2), "Quantizing transformer") as progress_bar:
    #
    #         # progress_bar.update()
    #         # progress_bar.set_description(f"quantizing flux_transformer to qfloat8")
    #
    #         # if not fill:
    #
    #         quantize(self.pipe.transformer, weights=qfloat8)
    #         progress_bar.update()
    #         progress_bar.set_description(f"freezing flux_transformer")
    #         # if not fill:
    #         freeze(self.pipe.transformer)
    #         progress_bar.update()
    #
    #     # # load and quantize t5
    #     # with tqdm(total=3, desc="loading text_encoder_2") as progress_bar:
    #     #
    #     #     progress_bar.update()
    #     #     # progress_bar.set_description("quantizing text_encoder_2 to qfloat8")
    #     #     # if not fill:
    #     #     #     quantize(self.text_encoder_2, weights=qfloat8)
    #     #     progress_bar.update()
    #     #     # progress_bar.set_description("freezing text_encoder_2")
    #     #     # if not fill:
    #     #     #     freeze(self.text_encoder_2)
    #     #     progress_bar.update()

    def apply_flux_loras_with_prompt(self, prompt, use_turbo=False):
        # use_turbo = False
        self.loaded_loras = {}
        self.pipe.unload_lora_weights()
        # if use_turbo:
        #     prompt += "<turbo-a>"
        filtered_prompt, loras = self.extract_lora_params_from_prompt(prompt)
        if len(loras) == 0:
            # set
            active_adapters = self.pipe.get_active_adapters()
            print("no loras detected in prompt. setting adapters:", active_adapters)
            return filtered_prompt

        # return filtered_prompt

        loras_filtered_names = {}
        # filter the names for periods or will throw an error when loading the lora
        for adapter_name, adapter_scale in loras.items():
            adapter_name_filtered_for_periods = adapter_name.replace(".", "dot")
            adapter_name_filtered_for_periods = adapter_name_filtered_for_periods.replace("/", "slash")
            loras_filtered_names[adapter_name] = adapter_name_filtered_for_periods

        for adapter_name, adapter_scale in loras.items():
            adapter_name_filtered_for_periods = loras_filtered_names[adapter_name]
            # load the lora into pipeline
            lora_file = os.path.join(self.lora_path, f"{adapter_name}.safetensors")
            if os.path.exists(lora_file):
                self.pipe.load_lora_weights(self.lora_path,
                                          weight_name=f"{adapter_name}.safetensors",
                                          adapter_name=adapter_name_filtered_for_periods)
                print("Loaded lora into pipe from", lora_file)
            else:
                # try to download from huggingface
                try:
                    self.pipe.load_lora_weights(
                        "Anyfusion/flux-loras",
                        weight_name=f"{adapter_name}.safetensors",
                        adapter_name=adapter_name_filtered_for_periods)
                    print("Loaded lora into pipe from", lora_file)
                except huggingface_hub.errors.RepositoryNotFoundError:
                    print("WARNING:", f"{adapter_name}.safetensors", adapter_name_filtered_for_periods, "does not exist.", "Ignoring.")
                    continue
                except Exception as e:
                    print("WARNING:", str(e), "Ignoring.")
                    continue

            self.loaded_loras[adapter_name_filtered_for_periods] = float(adapter_scale)
        if len(self.loaded_loras) > 0:
            lora_settings = {
                "adapter_names": list(self.loaded_loras.keys()),
                "adapter_weights": list(self.loaded_loras.values())
            }
            self.pipe.set_adapters(**lora_settings)
            active_adapters = self.pipe.get_active_adapters()
            print("loras set:", active_adapters)
        else:
            print("loras set:", None)
        return filtered_prompt

    def extract_lora_params_from_prompt(self, prompt):
        assert isinstance(prompt, str)

        loras = {}
        open_bracket, closed_bracket = "<", ">"
        prompt_dict = {"prompt": prompt, "current_prompt":prompt}
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
                        adapter_name, adapter_scale = self.create_lora_params_from_prompt(prompt_dict["current_prompt"], open_bracket_idx, closed_bracket_idx)
                        if adapter_name is not None and adapter_name not in loras.keys():
                            loras[adapter_name] = adapter_scale
                        lora_string = prompt_dict["current_prompt"][open_bracket_idx:closed_bracket_idx+1]
                        replaced = prompt_dict["current_prompt"].replace(lora_string, "")
                        prompt_dict["current_prompt"] = replaced
                        break
                if idx >= len(prompt_dict["current_prompt"])-1:
                    eos = True
                    break
            if eos:
                break

        return prompt_dict["current_prompt"], loras

    def create_lora_params_from_prompt(self, prompt, open_bracket_idx, closed_bracket_idx):
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


class ImageResizeParams:
    # contains the modified h and w along with the crop coords in x1x2y1y2
    def __init__(self, h, w, max_side, center_crop):
        self.center_crop = center_crop
        if center_crop:
            new_w = new_h = min(w, h)
            self.xmin = int((max(w, h) - h) / 2)
            self.xmax = self.xmin + new_w
            self.ymin = int((max(w, h) - w) / 2)
            self.ymax = self.ymin + new_w
            print("new_w", new_w, "new_h", new_h, "x1x2y1y2", self.xmin, self.xmax, self.ymin, self.ymax)

            self.new_w = self.new_h = max_side
        else:
            ratio = max(w, h) / max_side
            new_w = int(w / ratio)
            new_h = int(h / ratio)
            # sd inputs must be divisible by 8
            self.new_w = new_w - (new_w % 8)
            self.new_h = new_h - (new_h % 8)
            assert new_w > 0 and new_h > 0
            print("new_w", new_w, "new_h", new_h)

    def apply_params(self, image):
        if isinstance(image, Image.Image):
            if self.center_crop:
                image = np.array(image)
                image = image[self.ymin:self.ymax, self.xmin:self.xmax, :]
                image = cv2.resize(image, (self.new_w, self.new_h), interpolation=cv2.INTER_CUBIC)
                image = Image.fromarray(image)
            else:
                image.resize((self.new_w, self.new_h))
        elif isinstance(image, np.ndarray):
            if self.center_crop:
                image = image[self.ymin:self.ymax, self.xmin:self.xmax, :]
                image = cv2.resize(image, (self.new_w, self.new_h), interpolation=cv2.INTER_CUBIC)
            else:
                image = cv2.resize(image, (self.new_w, self.new_h), interpolation=cv2.INTER_CUBIC)

        return image
