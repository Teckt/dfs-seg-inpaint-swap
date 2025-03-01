import os
import random
import time


import torch

from pipe_sd15 import SD15PipelineManager


from redresser_flux import Redresser, ImageGenerator
from redresser_utils import RedresserSettings


class RedresserSD15(Redresser):

    def __init__(self, is_server=False, local_files_only=True):
        super().__init__(is_server, local_files_only, model="sd15-fill")
        '''
        ignore this
        settings(dict): run with redresser and pipe settings
            -default pipe settings (will only update these keys if you have them)
            "mode": SD15PipelineManager.USE_IMAGE (0),
            "use_LCM": True, # only used for video pipe
            "scheduler": "EulerAncestralDiscreteScheduler",
            "use_inpaint_control_net": True,
            "control_net_id": 'openpose'
            -default redresser settings
        '''
        # initialize the sd pipeline
        self.pipe = SD15PipelineManager(local_files_only=local_files_only)
        # update settings
        pipe_settings = {
            "mode": SD15PipelineManager.IMAGE_MODE,  # don't change this
            "use_LCM": True,  # only used for video pipe
            "scheduler": "EulerAncestralDiscreteScheduler",  # can change
            "use_inpaint_control_net": True,  # don't change this
            "control_net_id": 'openpose'  # don't change this
        }
        self.pipe.pipe_settings = pipe_settings
        self.pipe.set_pipeline()

        self.settings = RedresserSettings()

    def run(self, batch_frames, orig_imgs, seg_imgs, control_images_inpaint, control_images_p, yolo_results, image_resize_params, image_path):

        seed = self.settings.options.get("seed", -1)
        if seed is None or seed < 0:
            seed = int(time.time())

        print("seed", seed)
        generator = torch.Generator().manual_seed(seed)

        # filter and extract loras from prompt and apply them to the pipe
        prompt = self.settings.options["prompt"]
        if self.model == "flux":
            prompt = self.pipe.apply_flux_loras_with_prompt(prompt)

        args = {
            "prompt": prompt,
            "image": orig_imgs[0],
            "mask_image": seg_imgs[0],
            "height": image_resize_params.new_h,
            "width": image_resize_params.new_w,
            "guidance_scale": random.uniform(3.0, 5.5),  # self.settings.options["guidance_scale"],
            "num_inference_steps": 30,  # self.settings.options["num_inference_steps"],
            # "max_sequence_length": 512,
            # "generator": torch.Generator("cpu").manual_seed(88)
        }

        if self.pipe.pipe_settings.get("use_inpaint_control_net", True):
            args["control_image"] = control_images_inpaint[0]#[control_images_inpaint, control_images_p]
        else:
            args["image"] = control_images_p[0]

        if self.settings.options.get("negative_prompt", "") != "":
            args["negative_prompt"] = self.settings.options["negative_prompt"]
        if int(self.settings.options.get("strength", 0)) > 0:
            args["strength"] = int(self.settings.options["strength"])
        if int(self.settings.options.get("clip_skip", 0)) > 0:
            args["clip_skip"] = int(self.settings.options["clip_skip"])
        if generator is not None:
            args["generator"] = generator

        with torch.inference_mode():
            output = self.pipe.pipe(
                **args
            )

        final_pil_images = self.process_outputs(output.images, yolo_results)

        time_id = time.time()
        if self.is_server:
            basename = os.path.basename(image_path)
            redresser_dir = image_path.replace(basename, "")
            redresser_output_file_path = f"{redresser_dir}/outputImage.png"
        else:
            # save to same dir as image
            basename = os.path.basename(image_path)
            redresser_slug = "redressed_results"
            redresser_dir = image_path.replace(basename, redresser_slug)
            redresser_output_file_path = f"{redresser_dir}/{seed}-{self.settings.options['guidance_scale']}-{self.settings.options['num_inference_steps']}-{basename}"
            if not os.path.exists(redresser_dir):
                os.mkdir(redresser_dir)

        for image_idx, image in enumerate(final_pil_images):
            print("image_idx", image_idx, "image", image, image.info)
            image.save(redresser_output_file_path)
            # image.save(f"{time_id}_{str(image_idx).zfill(5)}.png")


class ImageGeneratorSD15(ImageGenerator):
    def __init__(self, is_server=False, local_files_only=True):
        super().__init__(is_server, local_files_only, model="sd15")
        '''
        ignore this
        settings(dict): run with redresser and pipe settings
            -default pipe settings (will only update these keys if you have them)
            "mode": SD15PipelineManager.USE_IMAGE (0),
            "use_LCM": True, # only used for video pipe
            "scheduler": "EulerAncestralDiscreteScheduler",
            "use_inpaint_control_net": True,
            "control_net_id": 'openpose'
            -default redresser settings
        '''
        # initialize the sd pipeline
        self.pipe = SD15PipelineManager(local_files_only=local_files_only)
        # update settings
        pipe_settings = {
            "mode": SD15PipelineManager.TEXT_MODE,  # don't change this
            "use_LCM": True,  # only used for video pipe
            "scheduler": "EulerAncestralDiscreteScheduler",  # can change
            "use_inpaint_control_net": True,  # don't change this
            "control_net_id": 'openpose'  # don't change this
        }
        self.pipe.pipe_settings = pipe_settings
        print("PIPE SETTINGS")
        print(self.pipe.pipe_settings)
        self.pipe.set_pipeline()

        self.settings = RedresserSettings()