import random
from multiprocessing import Queue, Process

import os
import time

import cv2
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from tf_free_functions import paste_swapped_image

# from pipe_manager import SD15PipelineManager

from redresser_utils import RedresserSettings, ImageResizeParams, yolo8_extract_faces, yolo_segment_image, make_inpaint_condition

import numpy as np

from flux_pipe import MyFluxPipe


class Redresser:

    def __init__(self, is_server=False, local_files_only=True, model="flux-fill"):
        '''

        :param is_server: determines the output name in redresser_output_file_path
        :param local_files_only:
        '''
        self.is_server = is_server
        self.model = model
        # initialize default settings first
        self.settings = RedresserSettings()

        if self.model == "flux-fill":
            # initialize the sd pipeline
            self.pipe = MyFluxPipe()
        else:
            self.pipe = None

    def run(self, batch_frames, orig_imgs, seg_imgs, control_images_inpaint, control_images_p, yolo_results, image_resize_params, image_path):

        seed = self.settings.options.get("seed", -1)
        if seed is None or seed < 0:
            seed = int(time.time())

        print("seed", seed)
        generator = torch.Generator().manual_seed(seed)

        # filter and extract loras from prompt and apply them to the pipe
        prompt = self.settings.options["prompt"]
        if self.model == "flux-fill":
            prompt = self.pipe.apply_flux_loras_with_prompt(prompt)

        args = {
            "prompt": prompt,
            "image": orig_imgs[0],
            "mask_image": seg_imgs[0],
            "height": image_resize_params.new_h,
            "width": image_resize_params.new_w,
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

    def process_outputs(self, outputs, yolo_results):
        final_pil_images = []
        final_cv2_images = []
        segment_id = self.settings.options["SEGMENT_ID"]
        for image_idx, image in enumerate(outputs):

            np_image = np.array(image)

            # paste each original face back one by one; skip if segmenting face
            if segment_id != RedresserSettings.SEGMENT_FACE and self.settings.options["keep_face"]:
                for (face_index, face_data) in yolo_results[image_idx].items():
                    # no keys = nothing extracted
                    if 'aligned_cropped_image' not in face_data.keys():
                        continue

                    swapped_image = face_data[
                        "aligned_cropped_image"]  # unsharp_mask(extracted_data["aligned_cropped_image"], amount=.5)
                    np_image = paste_swapped_image(
                        dst_image=np_image,
                        swapped_image=swapped_image,
                        seg_mask=face_data["seg_mask"],
                        aligned_cropped_params=face_data["aligned_cropped_params"],
                        seamless_clone=False,
                        blur_mask=True,
                        resize=False
                    )
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
            final_cv2_images.append(np_image)

            image = Image.fromarray(cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)).convert('RGB')
            final_pil_images.append(image)

        return final_pil_images


class ImageGenerator:
    def __init__(self, is_server=False, local_files_only=True, model="flux"):
        '''

        :param is_server: determines the output name in redresser_output_file_path
        :param local_files_only:
        '''
        self.is_server = is_server
        self.model = model
        # initialize default settings first
        self.settings = RedresserSettings()

        if self.model == "flux":
            # initialize the sd pipeline
            self.pipe = MyFluxPipe(fill=False)
        else:
            self.pipe = None

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

        args = {
            "prompt": prompt,
            "height": 1024,
            "width": 1024,
            "guidance_scale": random.uniform(3.5, 7.5),  # self.settings.options["guidance_scale"],
            "num_inference_steps": 8  # self.settings.options["num_inference_steps"],
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

        with torch.inference_mode():
            output = self.pipe.pipe(
                **args
            )

        final_pil_images = self.process_outputs(output.images)

        time_id = time.time()
        if self.is_server:
            basename = os.path.basename(image_path)
            redresser_dir = image_path.replace(basename, "")
            redresser_output_file_path = f"{redresser_dir}/outputImage.png"
        else:
            # save to same dir as image
            basename = os.path.basename(image_path)
            redresser_slug = "generated_results"
            redresser_dir = image_path.replace(basename, redresser_slug)
            redresser_output_file_path = f"{redresser_dir}/{seed}-{self.settings.options['guidance_scale']}-{self.settings.options['num_inference_steps']}-{basename}"
            if not os.path.exists(redresser_dir):
                os.mkdir(redresser_dir)

        for image_idx, image in enumerate(final_pil_images):
            print("image_idx", image_idx, "image", image, image.info)
            image.save(redresser_output_file_path)
            # image.save(f"{time_id}_{str(image_idx).zfill(5)}.png")
        pass

    def process_outputs(self, outputs):
        final_pil_images = []
        final_cv2_images = []

        for image_idx, image in enumerate(outputs):

            np_image = np.array(image)
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
            final_cv2_images.append(np_image)

            image = Image.fromarray(cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)).convert('RGB')
            final_pil_images.append(image)

        return final_pil_images

def run_image_processor(input_queue: Queue, output_queue:Queue):
    gpu_index = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    from redresser_flux_im import ImageProcessor

    image_processor = ImageProcessor()
    while True:
        settings = input_queue.get(True)
        image_processor.set_seg_models(settings)

        if os.path.isfile(settings.options['image']):
            output_queue.put(image_processor.prepare_inputs(
                settings.options['image'],
                settings.options["max_side"],
                settings.options["center_crop"]))
        else:
            image_dir = settings.options['image']
            for file in os.listdir(image_dir):
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".webp") or file.endswith(".jfif"):
                    image_path = f"{image_dir}/{file}"
                    print(f"Processing {image_path}")
                    try:
                        output_queue.put(image_processor.prepare_inputs(
                            image_path,
                            settings.options["max_side"],
                            settings.options["center_crop"]))
                    except Exception as e:
                        print(e)

        output_queue.put(None, block=True)



if __name__ == "__main__":

    batch_size = 1  # since we're not doing video, just process 1 image
    r = Redresser()
    input_queue = Queue(1)
    output_queue = Queue(1)

    image_process = Process(
        target=run_image_processor,
        kwargs={
            "input_queue": input_queue,
            "output_queue": output_queue})
    image_process.start()

    while True:
        r.settings.set_options()
        input_queue.put(r.settings, block=True)

        while True:
            im_outputs = output_queue.get(block=True)
            if im_outputs is None:
                break
            r.run(*im_outputs)
