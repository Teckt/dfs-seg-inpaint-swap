import random
import sys
from multiprocessing import Queue, Process

import os
import time

import cv2
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from CONSTANTS import *
from fire_functions import FirestoreFunctions
from tf_free_functions import paste_swapped_image, paste_image_with_mask

# from pipe_manager import SD15PipelineManager

from redresser_utils import RedresserSettings, ImageResizeParams, yolo8_extract_faces, yolo_segment_image, \
    make_inpaint_condition, load_image

import numpy as np

from flux_pipe import MyFluxPipe


class Redresser:
    total_steps = 8  # a hack for now for the end of steps callback

    def __init__(self, is_server=False, local_files_only=True, model="flux-fill", use_hyper=False, use_kontext=False):
        '''

        :param is_server: determines the output name in redresser_output_file_path
        :param local_files_only:
        '''
        self.use_kontext = use_kontext
        self.use_hyper = use_hyper  # uses the hyper flux nf4 path instead of the normal path
        self.is_server = is_server
        self.model = model
        # initialize default settings first
        self.settings = RedresserSettings()

        if self.use_kontext:
            self.pipe_manager = MyFluxPipe(fill=True, use_hyper=self.use_hyper, use_kontext=use_kontext)
        elif self.model == "fill":
            # initialize the sd pipeline
            self.pipe_manager = MyFluxPipe(fill=True, use_hyper=self.use_hyper)
        else:
            self.pipe_manager = MyFluxPipe(fill=False, use_hyper=self.use_hyper)

    def switch_pipeline(self, model):
        """
        changes is_fill property of fluxpipe and sets a new flux_model_name
        :param model:
        :return:
        """
        self.model = model
        self.pipe_manager.switch_pipe(fill=model == "fill")

    def parse_image_processor_outputs(self, batch_frames, orig_paths, seg_paths, control_images_inpaint, control_images_p, yolo_results_condensed, image_resize_params, image_path):
        """
        :param batch_frames: None
        :param orig_imgs: list of image file paths, load images and convert to PIL
        :param seg_imgs: list of image file paths, load images and convert to PIL
        :param control_images_inpaint: None; do the inpaint condition here
        :param control_images_p: None
        :param yolo_results: converted and condensed dict, load images and convert to numpy
        :param image_resize_params: None
        :param image_path: same

        """
        orig_imgs = [load_image(i, "pil") for i in orig_paths]
        seg_imgs = [load_image(i, "pil") for i in seg_paths]

        control_images_inpaint = [
            make_inpaint_condition(orig_imgs[i], seg_imgs[i], 0.5)
            for i in range(len(orig_imgs))
        ]
        yolo_results = {}
        for image_index, yolo_result in yolo_results_condensed.items():
            if image_index not in yolo_results.keys():
                yolo_results[image_index] = {}

            for (face_index, face_data) in yolo_result.items():
                if face_index not in yolo_results[image_index].keys():
                    yolo_results[image_index][face_index] = {}
                if 'aligned_cropped_image' in face_data.keys():
                    aligned_cropped_image = load_image(face_data['aligned_cropped_image_path'], "numpy")
                    seg_mask = load_image(face_data['seg_mask_path'], "numpy")
                    yolo_results[image_index][face_index]['aligned_cropped_params'] = face_data[ 'aligned_cropped_params']
                    yolo_results[image_index][face_index]['aligned_cropped_image'] = aligned_cropped_image
                    yolo_results[image_index][face_index]['seg_mask'] = seg_mask

        return batch_frames, orig_imgs, seg_imgs, control_images_inpaint, control_images_p, yolo_results, image_resize_params, image_path

    def run(self, batch_frames, orig_imgs, seg_imgs, control_images_inpaint, control_images_p, yolo_results, image_resize_params, image_path):
        mode = int(self.settings.options["mode"])

        if mode == 0:
            # switch to t2i
            if self.model == 'fill':  # fill
                self.switch_pipeline('t2i')
            self.run_t2i()
        else:
            # switch to fill
            if self.model == 't2i':  # t2i
                self.switch_pipeline('fill')
            if self.use_kontext:
                self.run_kontext()
            else:
                self.run_fill(batch_frames, orig_imgs, seg_imgs, control_images_inpaint, control_images_p, yolo_results,
                              image_resize_params, image_path)

    def run_kontext(self):
        orig_imgs = []
        image_path = self.settings.options.get("image")

        image = Image.open(image_path)
        orig_imgs.append(image)

        seed = self.settings.options.get("seed", -1)
        if seed is None or seed < 0:
            seed = int(time.time())

        print("seed", seed)
        generator = torch.Generator().manual_seed(seed)

        # filter and extract loras from prompt and apply them to the pipe
        prompt = self.settings.options["prompt"]
        prompt = self.pipe_manager.apply_flux_loras_with_prompt(prompt)

        # set to 8 if current pipeline is set up to use hyper
        if self.pipe_manager.use_hyper:
            self.settings.options["num_inference_steps"] = 8

        width, height = orig_imgs[0].size
        args = {
            "prompt": prompt,
            "image": orig_imgs[0],
            # "height": height,
            # "width": width,
            "guidance_scale": self.settings.options["guidance_scale"],
            "num_inference_steps": self.settings.options["num_inference_steps"],
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

        if self.is_server:
            Redresser.total_steps = args["num_inference_steps"]
            args["callback_on_step_end"] = update_progress

        with torch.inference_mode():
            output = self.pipe_manager.pipe(
                **args
            )

        final_pil_images = self.process_t2i_outputs(output.images)

        time_id = time.time()
        if self.is_server:
            basename = os.path.basename(image_path)
            redresser_dir = image_path.replace(basename, "")
            redresser_output_file_path = f"{redresser_dir}/{OUTPUT_FILE_BASE_NAME}"
        else:
            def dict_to_filename(data: dict, separator="-", kv_separator="_"):
                return separator.join(f"{k}{kv_separator}{v}" for k, v in data.items())

            # save to same dir as image
            redresser_output_file_path = f"{IMAGE_OUTPUT_DIR}/{seed}-{self.settings.options['guidance_scale']}-{self.settings.options['num_inference_steps']}-{dict_to_filename(self.pipe_manager.loaded_loras)}.png"
            if not os.path.exists(IMAGE_OUTPUT_DIR):
                os.mkdir(IMAGE_OUTPUT_DIR)

        for image_idx, image in enumerate(final_pil_images):
            print("image_idx", image_idx, "image", image, image.info)
            image.save(redresser_output_file_path)
            # image.save(f"{time_id}_{str(image_idx).zfill(5)}.png")

    def run_fill(self, batch_frames, orig_imgs, seg_imgs, control_images_inpaint, control_images_p, yolo_results, image_resize_params, image_path):

        seed = self.settings.options.get("seed", -1)
        if seed is None or seed < 0:
            seed = int(time.time())

        print("seed", seed)
        generator = torch.Generator().manual_seed(seed)

        # filter and extract loras from prompt and apply them to the pipe
        prompt = self.settings.options["prompt"]
        prompt = self.pipe_manager.apply_flux_loras_with_prompt(prompt)

        # set to 8 if current pipeline is set up to use hyper
        if self.pipe_manager.use_hyper:
            self.settings.options["num_inference_steps"] = 8

        width, height = orig_imgs[0].size
        args = {
            "prompt": prompt,
            "image": orig_imgs[0],
            "mask_image": seg_imgs[0],
            "height": height,
            "width": width,
            "guidance_scale": self.settings.options["guidance_scale"],
            "num_inference_steps": self.settings.options["num_inference_steps"],
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

        if self.is_server:
            Redresser.total_steps = args["num_inference_steps"]
            args["callback_on_step_end"] = update_progress

        with torch.inference_mode():
            output = self.pipe_manager.pipe(
                **args
            )

        final_pil_images = self.process_fill_outputs(orig_imgs, seg_imgs, output.images, yolo_results)

        time_id = time.time()
        if self.is_server:
            basename = os.path.basename(image_path)
            redresser_dir = image_path.replace(basename, "")
            redresser_output_file_path = f"{redresser_dir}/{OUTPUT_FILE_BASE_NAME}"

            redresser_output_mask_file_path = f"{redresser_dir}/{OUTPUT_MASK_FILE_BASE_NAME}"

            redresser_output_original_file_path = f"{redresser_dir}/{OUTPUT_ORIGINAL_FILE_BASE_NAME}"
        else:

            if os.path.exists(image_path):
                # save to same dir as image
                basename = os.path.basename(image_path)
                redresser_dir = image_path.replace(basename, REPAINT_OUTPUT_DIR)
            else:
                redresser_dir = REPAINT_OUTPUT_DIR

            redresser_output_file_path = f"{redresser_dir}/{seed}-{self.settings.options['guidance_scale']}-{self.settings.options['num_inference_steps']}-{OUTPUT_FILE_BASE_NAME}"
            redresser_output_mask_file_path = f"{redresser_dir}/{seed}-{self.settings.options['guidance_scale']}-{self.settings.options['num_inference_steps']}-{OUTPUT_MASK_FILE_BASE_NAME}"
            redresser_output_original_file_path = None
            if not os.path.exists(redresser_dir):
                os.mkdir(redresser_dir)

        for image_idx, image in enumerate(final_pil_images):
            print("image_idx", image_idx, "image", image, image.info)
            image.save(redresser_output_file_path)

            mask = seg_imgs[image_idx]

            np_mask = np.array(mask)
            np_mask = np.expand_dims(np_mask, -1)
            np_mask = np.concatenate((np_mask[:, :, 0], np_mask[:, :, 0]), axis=-1)
            np_mask = Image.fromarray(np_mask)
            np_mask.save(redresser_output_mask_file_path)

            if redresser_output_original_file_path is not None:
                original = orig_imgs[image_idx]
                original.save(redresser_output_original_file_path)
            # image.save(f"{time_id}_{str(image_idx).zfill(5)}.png")

    def run_t2i(self):
        image_path = self.settings.options.get("image")

        seed = self.settings.options.get("seed", -1)
        if seed is None or seed < 0:
            seed = int(time.time())

        print("seed", seed)
        generator = torch.Generator().manual_seed(seed)

        # filter and extract loras from prompt and apply them to the pipe
        prompt = self.settings.options["prompt"]
        prompt = self.pipe_manager.apply_flux_loras_with_prompt(prompt)

        height = self.settings.options.get("height", self.settings.options["height"])
        width = self.settings.options.get("width", self.settings.options["width"])

        args = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "guidance_scale": self.settings.options["guidance_scale"],
            "num_inference_steps": self.settings.options["num_inference_steps"],
            # "guidance_scale": random.uniform(3.5, 7.5),  # self.settings.options["guidance_scale"],
            # "num_inference_steps": 8  # self.settings.options["num_inference_steps"],
            # "max_sequence_length": 512,
            # "generator": torch.Generator("cpu").manual_seed(88)
        }

        if self.is_server:
            Redresser.total_steps = args["num_inference_steps"]
            args["callback_on_step_end"] = update_progress

        # if self.settings.options["negative_prompt"] is not None:
        #     args["negative_prompt"] = self.settings.options["negative_prompt"]
        # if self.settings.options["strength"] is not None:
        #     args["strength"] = self.settings.options["strength"]
        # if self.settings.options["clip_skip"] > 0:
        #     args["clip_skip"] = self.settings.options["clip_skip"]

        if generator is not None:
            args["generator"] = generator
        sys.stdout.flush()
        with torch.inference_mode():
            output = self.pipe_manager.pipe(
                **args
            )

        final_pil_images = self.process_t2i_outputs(output.images)

        time_id = time.time()
        if self.is_server:
            basename = os.path.basename(image_path)
            redresser_dir = image_path.replace(basename, "")
            redresser_output_file_path = f"{redresser_dir}/{OUTPUT_FILE_BASE_NAME}"
        else:
            def dict_to_filename(data: dict, separator="-", kv_separator="_"):
                return separator.join(f"{k}{kv_separator}{v}" for k, v in data.items())

            # save to same dir as image
            redresser_output_file_path = f"{IMAGE_OUTPUT_DIR}/{seed}-{self.settings.options['guidance_scale']}-{self.settings.options['num_inference_steps']}-{dict_to_filename(self.pipe_manager.loaded_loras)}.png"
            if not os.path.exists(IMAGE_OUTPUT_DIR):
                os.mkdir(IMAGE_OUTPUT_DIR)

        for image_idx, image in enumerate(final_pil_images):
            print("image_idx", image_idx, "image", image, image.info)
            image.save(redresser_output_file_path)
            # image.save(f"{time_id}_{str(image_idx).zfill(5)}.png")

    def process_t2i_outputs(self, outputs):
        final_pil_images = []
        final_cv2_images = []

        for image_idx, image in enumerate(outputs):
            np_image = np.array(image)
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
            final_cv2_images.append(np_image)

            image = Image.fromarray(cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)).convert('RGB')
            final_pil_images.append(image)

        return final_pil_images

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


def update_progress(pipeline, i, t, callback_kwargs):
    # print("i", i, "t", t,)
    FirestoreFunctions.repaintImageJobsRef.document(FirestoreFunctions.job_id).set(
        {"swappedFramesProgress": int(100*(i/Redresser.total_steps))}, merge=True
    )

    return callback_kwargs