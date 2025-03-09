import os
import time
from PIL import Image
import numpy as np
import cv2
import torch
from diffusers import FluxFillPipeline, AutoencoderKL, FluxTransformer2DModel, FluxPipeline
from huggingface_hub import hf_hub_download
# from diffusers.utils import load_image
from tqdm import tqdm
from transformers import T5EncoderModel, CLIPTextModel
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as BitsAndBytesConfig
from CONSTANTS import *
from optimum.quanto import freeze, qfloat8, quantize
# from optimum_quanto.optimum.quanto import QuantizedTransformersModel


# from yolomodel.redresser.flux_controlnet_inpainting.transformer_flux import FluxTransformer2DModel as FluxTransformer2DModel_alimama
# from yolomodel.redresser.flux_controlnet_inpainting.controlnet_flux import FluxControlNetModel
# from yolomodel.redresser.flux_controlnet_inpainting.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline


class MyFluxPipe:

    def __init__(self, fill=True):

        self.dtype = torch.bfloat16
        self.lora_path = LORA_PATH
        self.flux_model_name = FLUX_FILL_PATH if fill else FLUX_PATH
        self.flux_hyper_model_name = FLUX_FILL_HYPER_PATH if fill else FLUX_HYPER_PATH
        self.flux_transformer = None
        # load VAE
        # with tqdm(range(1), "Loading flux_vae"):
        #     self.flux_vae = AutoencoderKL.from_pretrained(
        #         self.flux_model_name, subfolder="vae",
        #         # "C:/Users/teckt/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/ae.safetensors"
        #         # "C:\\Users\\teckt\\.cache\\huggingface\\hub\\models--Kijai--flux-fp8\\snapshots\\e77f550e3fe8be226884d5944a40abdbe4735ff5\\flux1-dev-fp8.safetensors",
        #         torch_dtype=self.dtype, local_files_only=True)
        #     # self.flux_vae.to(self.dtype)

        self.fused_turbo = False
        self.is_fill = fill

        # load tokenizer
        with tqdm(range(1), "Loading CLIPTextModel") as progress_bar:
            self.clip_L_text_encoder = CLIPTextModel.from_pretrained(
                TEXT_ENCODER_CLIP_L_PATH,
                torch_dtype=self.dtype,
                local_files_only=USE_LOCAL_FILES)
            progress_bar.update()
            
        pipeline_args = {
            "pretrained_model_name_or_path": self.flux_model_name,
            "transformer": None,
            "text_encoder": self.clip_L_text_encoder,
            "text_encoder_2": None,
            "torch_dtype": self.dtype,
            "local_files_only": USE_LOCAL_FILES
        }
        if USE_BNB:
            self.load_bnb_transformer_text_encoder_2()
            pipeline_args["device_map"] = "auto"
        else:
            self.load_transformer_text_encoder_2()  # inits the transformer and text_encoder_2
        print("loading pipeline")
        if fill:
            print("loading pipeline")
            self.pipe = FluxFillPipeline.from_pretrained(**pipeline_args)
        else:
            self.pipe = FluxPipeline.from_pretrained(**pipeline_args)
        # print("adding transformer")
        # self.pipe.transformer = self.flux_transformer
        # print("adding t5 encoder")
        # self.pipe.text_encoder_2 = self.text_encoder_2

        if FUSE_HYPER:
            #
            if not os.path.exists(f"{self.flux_hyper_model_name}/diffusion_pytorch_model.safetensors"):
                # fuse lora before quantizing
                self.fuse_hyper_lora()
                #
                #save the model here
                with tqdm(range(3), "Saving transformer") as p:

                    p.desc = "converting transformer to fp8"
                    self.flux_transformer.to('cuda', dtype=torch.float8_e4m3fn)
                    p.update()

                    p.desc = "saving transformer"
                    self.flux_transformer.save_pretrained("flux-fp8", max_shard_size="16GB")
                    p.update()

                    p.desc = f"converting transformer back and dtype({self.dtype})"
                    self.flux_transformer.to(dtype=self.dtype)
                    p.update()

        # quantize
        if not USE_BNB and USE_OPTIMUM_QUANTO:
            self.quanto_quantize()

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

        if VRAM < 22:
            # Must use a cpu offload option if quantization is fp8 or higher for flux
            if USE_CPU_OFFLOAD:
                print("using cpu offload")
                self.pipe.enable_model_cpu_offload()
            else:
                print("using sequential cpu offload")
                self.pipe.enable_sequential_cpu_offload()
        else:
            if USE_CPU_OFFLOAD:
                print("using cpu offload")
                self.pipe.enable_model_cpu_offload()
            elif USE_SEQUENTIAL_CPU_OFFLOAD:
                # disable cpu offload if quantized for 24GB
                print("using sequential cpu offload")
                self.pipe.enable_sequential_cpu_offload()
            else:
                # must call to cuda if disabled cpu offload
                self.pipe.to("cuda")

    def fuse_hyper_lora(self):
        # control_pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        # control_pipe.load_lora_weights("black-forest-labs/FLUX.1-Depth-dev-lora", adapter_name="depth")

        if self.fused_turbo:
            print("turbo already fused")
            return
        with tqdm(range(4), desc="Fusing hyper lora") as p:
            self.pipe.load_lora_weights(
                hf_hub_download(FUSE_HYPER_REPO, FUSE_HYPER_MODEL_FILE),
                adapter_name="turbo"
            )

            # self.pipe.load_lora_weights(self.lora_path,
            #                             weight_name="turbo-a.safetensors",
            #                             adapter_name="turbo")
            p.update()
            p.desc = "Setting adapter"
            self.pipe.set_adapters(["turbo"], adapter_weights=[FUSE_HYPER_ALPHA])

            p.update()
            p.desc = "Fusing lora"
            self.pipe.fuse_lora()
            p.update()
            p.desc = "Unloading lora"
            self.pipe.unload_lora_weights()
            p.desc = "Deleting adapter"
            # self.pipe.delete_adapters(["hyper-sd"])
            self.pipe.delete_adapters(["turbo"])
            p.update()
        self.fused_turbo = True

    def load_bnb_transformer_text_encoder_2(self):

        with tqdm(range(1), "Loading and quantizing t5 encoder") as progress_bar:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            self.text_encoder_2 = T5EncoderModel.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                subfolder="text_encoder_2",
                quantization_config=quant_config,
                torch_dtype=self.dtype,
                local_files_only=USE_LOCAL_FILES
            )
            progress_bar.update()

        with tqdm(range(1), "Loading and quantizing transformer") as progress_bar:
            quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
            self.flux_transformer = FluxTransformer2DModel.from_pretrained(
                self.flux_model_name,
                subfolder="transformer",
                quantization_config=quant_config,
                torch_dtype=self.dtype,
                local_files_only=USE_LOCAL_FILES
            )
            progress_bar.update()

    def load_transformer_text_encoder_2(self):
        # load and quantize transformer

        with tqdm(range(4), "Loading transformer") as progress_bar:
            if FUSE_HYPER:
                if os.path.exists(self.flux_hyper_model_name):
                    model_name = self.flux_hyper_model_name
                else:
                    model_name = self.flux_model_name
            else:
                model_name = self.flux_model_name
            print(f"loading from {model_name}")
            if self.is_fill:
                self.flux_transformer = FluxTransformer2DModel.from_pretrained(
                    model_name, subfolder="transformer",
                    # "flux-fill-fp8", subfolder="transformer",
                    torch_dtype=self.dtype, local_files_only=True if FUSE_HYPER and os.path.exists(self.flux_hyper_model_name) else USE_LOCAL_FILES)
            else:
                # self.flux_transformer = FluxTransformer2DModel.from_single_file(
                #     "C:\\Users\\teckt\\.cache\\huggingface\\hub\\models--Kijai--flux-fp8\\snapshots\\e77f550e3fe8be226884d5944a40abdbe4735ff5\\flux1-dev-fp8.safetensors",
                #     subfolder="transformer",
                #     torch_dtype=self.dtype, local_files_only=True)
                self.flux_transformer = FluxTransformer2DModel.from_pretrained(
                    model_name, subfolder="transformer",
                    # "flux-fp8", subfolder="transformer",
                    torch_dtype=self.dtype, local_files_only=True if FUSE_HYPER and os.path.exists(self.flux_hyper_model_name) else USE_LOCAL_FILES)
            progress_bar.update()

            # if not self.is_fill:
            #     progress_bar.set_description(f"fusing turbo-alpha lora before quantization")
            #     self.fuse_turbo()
            progress_bar.update()
            # progress_bar.set_description(f"quantizing flux_transformer to qfloat8")

            # if not fill:

            # quantize(self.flux_transformer, weights=qfloat8)
            progress_bar.update()
            # progress_bar.set_description(f"freezing flux_transformer")
            # if not fill:
            # freeze(self.flux_transformer)
            progress_bar.update()

            # self.flux_transformer.to(self.dtype)

        # load and quantize t5
        with tqdm(total=1, desc="loading text_encoder_2") as progress_bar:
            self.text_encoder_2 = T5EncoderModel.from_pretrained(
                FLUX_PATH,
                subfolder="text_encoder_2",
                torch_dtype=self.dtype, local_files_only=USE_LOCAL_FILES)
            progress_bar.update()
            # progress_bar.set_description("quantizing text_encoder_2 to qfloat8")
            # # if not fill:
            # quantize(self.text_encoder_2, weights=qfloat8)
            # progress_bar.update()
            # progress_bar.set_description("freezing text_encoder_2")
            # # if not fill:
            # freeze(self.text_encoder_2)
            # progress_bar.update()

    def quanto_quantize(self):
        # load and quantize transformer
        with tqdm(range(2), "Quantizing transformer") as progress_bar:

            # progress_bar.update()
            # progress_bar.set_description(f"quantizing flux_transformer to qfloat8")

            # if not fill:

            quantize(self.flux_transformer, weights=qfloat8)
            progress_bar.update()
            progress_bar.set_description(f"freezing flux_transformer")
            # if not fill:
            freeze(self.flux_transformer)
            progress_bar.update()

            # self.flux_transformer.to(self.dtype)

        # # load and quantize t5
        # with tqdm(total=3, desc="loading text_encoder_2") as progress_bar:
        #
        #     progress_bar.update()
        #     # progress_bar.set_description("quantizing text_encoder_2 to qfloat8")
        #     # if not fill:
        #     #     quantize(self.text_encoder_2, weights=qfloat8)
        #     progress_bar.update()
        #     # progress_bar.set_description("freezing text_encoder_2")
        #     # if not fill:
        #     #     freeze(self.text_encoder_2)
        #     progress_bar.update()

    def fuse_turbo(self):
        if self.fused_turbo:
            print("turbo already fused")
            return
        with tqdm(range(4), desc="Fusing turbo alpha lora") as p:
            self.flux_transformer.load_lora_adapter(self.lora_path,
                                        weight_name="turbo-a.safetensors",
                                        adapter_name="turbo")
            p.update()
            p.desc = "Setting adapter"
            lora_settings = {"adapter_names": ["turbo"],
                             "weights": [1.0]}
            self.flux_transformer.set_adapters(**lora_settings)
            p.update()
            p.desc = "Fusing lora"
            self.flux_transformer.fuse_lora()
            p.update()
            p.desc = "Unloading lora"
            self.flux_transformer.unload_lora()
            p.desc = "Deleting adapter"
            self.flux_transformer.delete_adapters(["turbo"])
            p.update()
        self.fused_turbo = True

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
            loras_filtered_names[adapter_name] = adapter_name_filtered_for_periods

        for adapter_name, adapter_scale in loras.items():
            adapter_name_filtered_for_periods = loras_filtered_names[adapter_name]
            # load the lora into pipeline
            lora_file = os.path.join(self.lora_path, f"{adapter_name}.safetensors")
            if os.path.exists(lora_file):
                print("Loading lora into pipe from", lora_file)
                self.pipe.load_lora_weights(self.lora_path,
                                          weight_name=f"{adapter_name}.safetensors",
                                          adapter_name=adapter_name_filtered_for_periods)
            else:
                print("WARNING:", lora_file, "does not exist.", "Ignoring.")
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
