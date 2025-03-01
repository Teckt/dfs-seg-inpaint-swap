import os
import time
from PIL import Image
import numpy as np
import cv2
import torch
from diffusers import FluxFillPipeline, AutoencoderKL, FluxTransformer2DModel, FluxPipeline
# from diffusers.utils import load_image
from tqdm import tqdm
from transformers import T5EncoderModel, CLIPTextModel
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as BitsAndBytesConfig

from optimum_quanto.optimum.quanto import freeze, qfloat8, quantize

# from yolomodel.redresser.flux_controlnet_inpainting.transformer_flux import FluxTransformer2DModel as FluxTransformer2DModel_alimama
# from yolomodel.redresser.flux_controlnet_inpainting.controlnet_flux import FluxControlNetModel
# from yolomodel.redresser.flux_controlnet_inpainting.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline


class MyFluxPipe:

    def __init__(self, fill=True):

        self.dtype = torch.bfloat16
        self.lora_path = "C:\\Users\\teckt\\PycharmProjects\\kohya\\kohya_ss\\training_data\\model"
        self.flux_model_name = "black-forest-labs/FLUX.1-Fill-dev" if fill else "black-forest-labs/FLUX.1-dev"
        # load VAE
        # with tqdm(range(1), "Loading flux_vae"):
        #     self.flux_vae = AutoencoderKL.from_pretrained(
        #         self.flux_model_name, subfolder="vae",
        #         # "C:/Users/teckt/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/ae.safetensors"
        #         # "C:\\Users\\teckt\\.cache\\huggingface\\hub\\models--Kijai--flux-fp8\\snapshots\\e77f550e3fe8be226884d5944a40abdbe4735ff5\\flux1-dev-fp8.safetensors",
        #         torch_dtype=self.dtype, local_files_only=True)
        #     # self.flux_vae.to(self.dtype)

        use_bnb = False
        # load tokenizer
        with tqdm(range(1), "Loading CLIPTextModel") as progress_bar:
            self.clip_L_text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=self.dtype,
                local_files_only=True)
            progress_bar.update()

        if use_bnb:
            with tqdm(range(1), "Loading and quantizing t5 encoder") as progress_bar:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                self.text_encoder_2 = T5EncoderModel.from_pretrained(
                    "black-forest-labs/FLUX.1-dev",
                    subfolder="text_encoder_2",
                    quantization_config=quant_config,
                    torch_dtype=torch.float16,
                    local_files_only=True
                )
                progress_bar.update()

            with tqdm(range(1), "Loading and quantizing transformer") as progress_bar:
                quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
                self.flux_transformer = FluxTransformer2DModel.from_pretrained(
                    self.flux_model_name,
                    subfolder="transformer",
                    quantization_config=quant_config,
                    torch_dtype=torch.float16,
                    local_files_only=True
                )
                progress_bar.update()
        else:
            # load and quantize transformer
            with tqdm(range(3), "Loading transformer") as progress_bar:
                self.flux_transformer = FluxTransformer2DModel.from_pretrained(
                    self.flux_model_name,
                # self.flux_transformer = FluxTransformer2DModel.from_single_file(
                    # "C:\\Users\\teckt\\PycharmProjects\\sd-trainer\\diffusers\\examples\\dreambooth\\flux-fill-fp8",
                    # "C:\\Users\\teckt\\.cache\\huggingface\\hub\\models--Kijai--flux-fp8\\snapshots\\e77f550e3fe8be226884d5944a40abdbe4735ff5\\flux1-dev-fp8.safetensors",
                    # self.flux_model_name,
                    subfolder="transformer",
                    torch_dtype=self.dtype, local_files_only=True)
                progress_bar.update()
                progress_bar.set_description(f"quantizing flux_transformer to qfloat8")
                if not fill:
                    quantize(self.flux_transformer, weights=qfloat8)
                progress_bar.update()
                progress_bar.set_description(f"freezing flux_transformer")
                if not fill:
                    freeze(self.flux_transformer)
                progress_bar.update()

                # self.flux_transformer.to(self.dtype)

            # load and quantize t5
            with tqdm(total=3, desc="loading text_encoder_2") as progress_bar:
                self.text_encoder_2 = T5EncoderModel.from_pretrained(
                    "black-forest-labs/FLUX.1-dev",
                    subfolder="text_encoder_2",
                    torch_dtype=self.dtype, local_files_only=True)
                progress_bar.update()
                progress_bar.set_description("quantizing text_encoder_2 to qfloat8")
                if not fill:
                    quantize(self.text_encoder_2, weights=qfloat8)
                progress_bar.update()
                progress_bar.set_description("freezing text_encoder_2")
                if not fill:
                    freeze(self.text_encoder_2)
                progress_bar.update()
        if fill:
            self.pipe = FluxFillPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Fill-dev",
                # tokenizer=self.clip_L_tokenizer,
                transformer=self.flux_transformer,
                text_encoder=self.clip_L_text_encoder,
                text_encoder_2=self.text_encoder_2,
                # vae=self.flux_vae,
                torch_dtype=torch.float16,
                local_files_only=True
            )
        else:
            # load this pipeline to get the tokenizers
            # self.pipe = FluxFillPipeline.from_pretrained(
            #     "black-forest-labs/FLUX.1-Fill-dev",
            #     # tokenizer=self.clip_L_tokenizer,
            #     transformer=None,
            #     text_encoder=None,
            #     text_encoder_2=None,
            #     vae=self.flux_vae,
            #     torch_dtype=self.dtype,
            #     local_files_only=True
            # )
            # load main pipe and transfer tokenizers to it
            print("loading pipeline")
            self.pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                transformer=None,
                text_encoder=self.clip_L_text_encoder,
                text_encoder_2=None,
                # vae=self.flux_vae,
                torch_dtype=self.dtype,
                # device_map="balanced",
                local_files_only=True
            )

            print("adding transformer")
            self.pipe.transformer = self.flux_transformer
            print("adding t5 encoder")
            self.pipe.text_encoder_2 = self.text_encoder_2

        # self.load_loras()

        # with tqdm(range(1), "Casting to float16"):
        #     self.pipe.to(torch.float16)
        # self.pipe.enable_free_noise_split_inference()

        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

    def load_loras(self):
        # return
        # loras
        self.pipe.load_lora_weights(
            self.lora_path, weight_name="jjk.safetensors", adapter_name="jjk")

        # self.pipe.load_lora_weights(self.lora_path,
        #                             weight_name="hyper8.safetensors",
        #                             adapter_name="turbo")
        #
        # print("converting lora to bfloat16")
        # self.pipe.to(torch.bfloat16)

        lora_settings = {
            # "adapter_names": ["jjk", "turbo"],
            "adapter_names": ["jjk"],
            # "adapter_weights": [0.875, 0.125]
            "adapter_weights": [1.0]
        }

        self.pipe.set_adapters(**lora_settings)
        # self.pipe.fuse_lora()

    def apply_flux_loras_with_prompt(self, prompt, use_turbo=False):
        self.loaded_loras = {}
        self.pipe.unload_lora_weights()
        if use_turbo:
            prompt += "<turbo-a>"
        filtered_prompt, loras = self.extract_lora_params_from_prompt(prompt)
        if len(loras) == 0:
            # set
            active_adapters = self.pipe.get_active_adapters()
            print("no loras detected in prompt. setting adapters:", active_adapters)
            return filtered_prompt

        return filtered_prompt

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


class FluxInputOptions:
    default_options = {
        "prompt": "jjk",
        # "image": "bb.jpg",
        # "mask": "seg/00000.png",
        "max_side": 1024,
        "guidance_scale": 6.0,
        "num_inference_steps": 20,
        "batch_size": 4,
        "seed": -1,
    }

    def __init__(self, ):
        self.options = {}
        self.previous_options = self.__class__.default_options.copy()

    def set_options(self):
        use_previous_for_rest = False
        for key, val in self.__class__.default_options.items():
            if use_previous_for_rest:
                self.options[key] = self.previous_options[key]
                continue

            self.options[key] = input(f'{key}({self.previous_options[key]}):')
            print("You've entered:", self.options[key])

            # PREVIOUS KEY
            if self.options[key] == '':
                self.options[key] = self.previous_options[key]
            # ALL PREVIOUS KEY
            elif self.options[key] == 'p':
                use_previous_for_rest = True
                print("using previous config:", self.previous_options)
                self.options[key] = self.previous_options[key]
            # RESET KEY
            elif self.options[key] == 'r':
                self.options[key] = val

        self.check_inputs()

        self.previous_options = self.options.copy()

    def check_inputs(self, ):
        for key, val in self.options.items():
            if key in ['guidance_scale', ]:
                try:
                    self.options[key] = float(self.options[key])
                except:
                    self.options[key] = self.__class__.default_options[key]
            if key in ['num_inference_steps', 'max_side', 'height', 'width', 'batch_size']:
                try:
                    self.options[key] = int(self.options[key])
                except:
                    self.options[key] = self.__class__.default_options[key]
            if key in ['use_dynamic_cfg', ]:
                try:
                    self.options[key] = bool(self.options[key])
                except:
                    self.options[key] = self.__class__.default_options[key]
            if key in ['image', 'mask']:
                if not os.path.exists(val):
                    self.options[key] = self.__class__.default_options[key]


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


if __name__ == "__main__":
    # gpu_index = 0
    # import os
    #
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    # import tensorflow as tf
    #
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)

    flux_pipe = MyFluxPipe(fill=False)
    input_options = FluxInputOptions()

    first_run = True
    while True:
        # get options
        input_options.set_options()
        # prepare image
        # image = Image.open(input_options.options["image"])
        # mask = Image.open(input_options.options["mask"])
        # image_resizer = ImageResizeParams(image.height, image.width, input_options.options["max_side"], False)
        # image = image_resizer.apply_params(image)
        # mask = image_resizer.apply_params(mask)
        # # create universal args
        # flux_args = {
        #     "prompt": input_options.options["prompt"],
        #     "image": image,
        #     "mask_image": mask,
        #     "height": image_resizer.new_h,
        #     "width": image_resizer.new_w,
        #     "guidance_scale": input_options.options["guidance_scale"],
        #     "num_inference_steps": input_options.options["num_inference_steps"],
        #     "max_sequence_length": 512,
        #     # "generator": torch.Generator("cpu").manual_seed(88)
        # }

        # normal
        flux_args = {
            "prompt": input_options.options["prompt"],
            "height": input_options.options["max_side"],
            "width": input_options.options["max_side"],
            "guidance_scale": input_options.options["guidance_scale"],
            "num_inference_steps": input_options.options["num_inference_steps"],
            # "max_sequence_length": 512,
            # "generator": torch.Generator("cpu").manual_seed(88)
        }

        # # modify if using alimama controlnet inpainting
        # if inpaint_type == "control":
        #     flux_args["control_image"] = flux_args.pop("image")
        #     flux_args["control_mask"] = flux_args.pop("mask_image")
        #     flux_args["negative_prompt"] = "bad quality, poor quality, blurry, grainy, artifacts"
        # run
        for i in range(input_options.options["batch_size"]):
            image = flux_pipe.pipe(**flux_args).images[0]
            # save
            image.save(f"{int(time.time())}-{input_options.options['num_inference_steps']}-{input_options.options['guidance_scale']}-flux-dev.png")
