import os
import pickle
import socket
import time

import torch
from huggingface_hub import hf_hub_download

from ultralytics import YOLO
from redresser_utils import RedresserSettings, ImageResizeParams, yolo8_extract_faces, yolo_segment_image, \
    make_inpaint_condition, pad_image, SocketClient, SocketServer
import cv2
import PIL.Image as Image
import numpy as np
from CONSTANTS import *

class ImageProcessor:
    def __init__(self):

        # builds face detection model from YOLO (PyTorch)
        print("loading face extract model")
        self.face_extract_model = YOLO(hf_hub_download("Anyfusion/xseg", RedresserSettings.face_model_path))

        self.f_seg_model = None
        self.hand_seg_model = None
        self.head_seg_model = None

        self.settings = None

    def set_seg_models(self, settings):
        self.settings = settings
        '''
        loads the models for face seg, face detect, fashion/person seg, hand seg
        '''
        segment_id = self.settings.options["SEGMENT_ID"]
        if segment_id not in [RedresserSettings.SEGMENT_FASHION, RedresserSettings.SEGMENT_PERSON,
                              RedresserSettings.SEGMENT_FACE, RedresserSettings.SEGMENT_ALL]:
            print("segment_type does not exist:", segment_id)
            assert False

        # skip if segmenting only face
        if segment_id == RedresserSettings.SEGMENT_FACE:
            if self.head_seg_model is None:
                print("loading head seg model")
                self.head_seg_model = YOLO(hf_hub_download("Bingsu/adetailer", RedresserSettings.face_box_model_path))
            return

        # segments images
        # if self.f_seg_model is None:
        print("loading f_seg model")
        if segment_id == RedresserSettings.SEGMENT_PERSON:
            self.f_seg_model = YOLO(hf_hub_download("Bingsu/adetailer", RedresserSettings.person_model_path))
        elif segment_id == RedresserSettings.SEGMENT_FASHION:
            self.f_seg_model = YOLO(hf_hub_download("Bingsu/adetailer", RedresserSettings.fashion_model_path))
            # else:
            #     print("Segment not implemented", segment_id)
            #     assert False

        if self.settings.options["keep_hands"]:
            if self.hand_seg_model is None:
                print("loading hand_seg model")
                self.hand_seg_model = YOLO(hf_hub_download("Bingsu/adetailer", RedresserSettings.hand_model_path))

    def prepare_inputs(self, image_path, mask_path, max_side, center_crop, seg_image_dir="seg"):
        if seg_image_dir is not None:
            if not os.path.exists(seg_image_dir):
                try:
                    os.mkdir(seg_image_dir)
                except Exception as e:
                    print(f"Error when making dir {seg_image_dir}", e)
            seg_path = f"{seg_image_dir}\\{os.path.basename(image_path)}"
        else:
            seg_path = f"{os.path.basename(image_path)}-seg.png" #{str(video_index).zfill(5)}

        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        if mask_path is not None:
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            mask = None

        padding_ratio = self.settings.options.get("padding", 0)
        # default 0%
        options_padding = int((h + w) * 0.5 * padding_ratio)
        if options_padding > 0:
            # paste image on white background with the padded borders
            # frame = pad_image(frame, options_padding)
            pad_outputs = pad_image(frame, mask, options_padding, use_noise=True)
            frame, pad_mask, mask = pad_outputs
            # frame, pad_mask, mask = pad_image(frame, mask, options_padding, use_noise=True)
            h, w = frame.shape[:2]

        image_resize_params = ImageResizeParams(
            h=h, w=w, max_side=max_side, center_crop=center_crop)
        frame = image_resize_params.apply_params(frame)
        if options_padding > 0:
            pad_mask = image_resize_params.apply_params(pad_mask)
        if mask is not None:
            mask = image_resize_params.apply_params(mask)

        print(f"processing image batch({0}) at {image_path}")

        batch_frames = {}
        orig_imgs = []
        seg_imgs = []
        control_images_inpaint = []
        for i in range(1):
            batch_frames[i] = frame

        # face segmentation
        yolo_results = yolo8_extract_faces(
            face_extractor=self.face_extract_model, face_seg_model=None, max_faces=10,
            conf_threshold=0.45, landmarks_padding_ratio=self.settings.options.get("face_mask_scale", 1.1),
            inputs=[frame for _ in batch_frames.items()],
            include_orig_img=False,
            # resizes outputs
            face_swapper_input_size=(256, 256)  # set to xseg output size so xseg doesn't resize again
        )
        try:
            face_masks_combined = yolo_results[0][0]['seg_mask_combined']
        except KeyError:
            face_masks_combined = np.zeros_like(frame)

        # redresser segmentation
        orig_img, seg_img, control_image_inpaint = self.process_image_for_segmentation(
            frame,
            mask,
            video_index=0,
            image_resize_params=image_resize_params,
            seg_path=seg_path,
            face_masks_combined=face_masks_combined,
            pad_mask=pad_mask if options_padding > 0 else None,
        )

        # batch_index = 0  # to correct indexing of the yolo results; but since it's an image, we don't need this because they're all the same so we just grab the first one
        for _frame_index, _ in batch_frames.items():
            orig_imgs.append(orig_img.copy())
            seg_imgs.append(seg_img.copy())
            control_images_inpaint.append(control_image_inpaint)
            # batch_index += 1
        print("images", len(orig_imgs), "control_images_inpaint", len(control_images_inpaint))

        # # We got errors with multi control net so skip this for now and only use inpaint control net
        # if isinstance(frame, Image.Image):
        #     frame = np.asarray(frame)
        # else:
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # control_images_p = [self.pipe_manager.control_p(frame)]
        control_images_p = None

        socket_outputs = self.prepare_inputs_for_socket(batch_frames, orig_imgs, seg_imgs, control_images_inpaint, control_images_p, yolo_results, image_resize_params, image_path)
        return socket_outputs

        # return batch_frames, orig_imgs, seg_imgs, control_images_inpaint, control_images_p, yolo_results, image_resize_params, image_path


    def prepare_inputs_for_socket(self, batch_frames, orig_imgs, seg_imgs, control_images_inpaint, control_images_p, yolo_results, image_resize_params, image_path):
        """

        :param batch_frames: list of numpy arrays of original frames
        :param orig_imgs: list of resized PIL images
        :param seg_imgs: list of resized PIL images
        :param control_images_inpaint: list of Torch tensors processed with make_inpaint_condition
        :param control_images_p: None
        :param yolo_results: dict batch_extracted_new_params from yolo extract
        :param image_resize_params: object to resize and crop images which also contains if center crop, orig hw
        :param image_path: file path to original image
        :returns
        :return batch_frames: None
        :return orig_imgs: list of image file paths
        :return seg_imgs: list of image file paths
        :return control_images_inpaint: None; pipe will do the inpaint condition
        :return control_images_p: None
        :return yolo_results: converted and condensed dict
        :return image_resize_params: None; nothing uses it so far
        :return image_path: file_path of image
        """
        if os.path.isdir(image_path):  # this should never be true
            assert False
            base_output_dir = image_path
        else:
            base_output_dir = image_path.replace(os.path.basename(image_path), "img_proc_outputs")
        if not os.path.exists(base_output_dir):
            os.mkdir(base_output_dir)
        run_output_dir = os.path.join(base_output_dir, str(int(time.time())))
        if not os.path.exists(run_output_dir):
            os.mkdir(run_output_dir)

        yolo_dict_for_dict = self.prepare_yolo_outputs_for_socket(yolo_results, run_output_dir)
        orig_img_paths = []
        for i, orig_img in enumerate(orig_imgs):
            orig_img_path = save_image(orig_imgs[0], os.path.join(run_output_dir, f"{i}_orig_img.png"))
            orig_img_paths.append(orig_img_path)
        seg_img_paths = []
        for i, seg_img in enumerate(seg_imgs):
            seg_img_path = save_image(seg_imgs[0], os.path.join(run_output_dir, f"{i}_seg_img.png"))
            seg_img_paths.append(seg_img_path)

        return None, orig_img_paths, seg_img_paths, None, None, yolo_dict_for_dict, None, image_path
    #     # save the orig_imgs
    #     # save the orig_imgs
    #
    #     # save the orig_imgs

    def prepare_yolo_outputs_for_socket(self, yolo_results, output_dir):
        """
        :param yolo_results: batch_extracted_new_params from yolo extract
        :param output_dir: the dir to save all the files (image index and face index in file name)
        :return:
        {  # example
            0: {  # image
                0: {  # face data
                    "aligned_cropped_image_path": img_path,
                    "seg_mask_path": seg_path,
                    "aligned_cropped_params": cropped_params
                },
            }
        }
        """

        socket_yolo_results = {}
        for image_index, yolo_result in yolo_results.items():
            if image_index not in socket_yolo_results.keys():
                socket_yolo_results[image_index] = {}

            for (face_index, face_data) in yolo_result.items():
                if face_index not in socket_yolo_results[image_index].keys():
                    socket_yolo_results[image_index][face_index] = {}
                aligned_cropped_image_path = save_image(
                    face_data['aligned_cropped_image'],
                    os.path.join(output_dir, f"yolo_{image_index}_{face_index}_aligned_cropped_image.png"))
                seg_mask_path = save_image(
                    face_data['seg_mask'],
                    os.path.join(output_dir, f"yolo_{image_index}_{face_index}_seg_mask.png"))
                socket_yolo_results[image_index][face_index]['aligned_cropped_image_path'] = aligned_cropped_image_path
                socket_yolo_results[image_index][face_index]['seg_mask_path'] = seg_mask_path
                socket_yolo_results[image_index][face_index]['aligned_cropped_params'] = face_data['aligned_cropped_params']
        return socket_yolo_results

    def process_image_for_segmentation(self, image, mask, video_index, image_resize_params, seg_path,
                                       face_masks_combined, pad_mask=None):
        '''
        seg_mask_combined(ndarray uint8): the combined segmented face masks for this image
        '''

        # image should be in RGB
        segment_id = self.settings.options["SEGMENT_ID"]
        if mask is not None:
            orig_img = Image.fromarray(image).convert("RGB")
            seg_img = Image.fromarray(mask).convert("RGB")
            orig_img.resize((image_resize_params.new_w, image_resize_params.new_h))
            seg_img.resize((image_resize_params.new_w, image_resize_params.new_h))
        elif segment_id == RedresserSettings.SEGMENT_FACE:

            # process and segment the image (person/fashion)
            orig_img, seg_img = yolo_segment_image(self.head_seg_model, image)[0]
            # _orig_img = np.array(orig_img, dtype=np.uint8)
            # seg_img = np.array(seg_img, dtype=np.uint8)
            # seg_img[_orig_img == 255] = 255

            seg_img = Image.fromarray(face_masks_combined)
            # seg_img = Image.fromarray(seg_img)

            orig_img = orig_img.resize((image_resize_params.new_w, image_resize_params.new_h))
            seg_img = seg_img.resize((image_resize_params.new_w, image_resize_params.new_h))
            return orig_img, seg_img, None
        elif segment_id == RedresserSettings.SEGMENT_ALL:
            orig_img = Image.fromarray(image).convert("RGB")
            # use all white as the mask
            seg_img = Image.new("RGB", (image_resize_params.new_w, image_resize_params.new_h), (255, 255, 255))
            orig_img.resize((image_resize_params.new_w, image_resize_params.new_h))
            seg_img.resize((image_resize_params.new_w, image_resize_params.new_h))
        else:
            # process and segment the image (person/fashion)
            orig_img, seg_img = yolo_segment_image(self.f_seg_model, image)[0]
            _orig_img = np.array(orig_img, dtype=np.uint8)
            seg_img = np.array(seg_img, dtype=np.uint8)
            seg_img[_orig_img == 255] = 255
            # test here to use noise on original where mask is
            # _orig_img[seg_img == 255] = np.random.randint(0, 256)
            seg_img = Image.fromarray(seg_img)
            # orig_img = Image.fromarray(_orig_img)
            orig_img = orig_img.resize((image_resize_params.new_w, image_resize_params.new_h))
        # if seg_path is not None:
        #     seg_img.save(seg_path)


        # process and segment the image (hands)
        if self.settings.options["keep_hands"]:
            _, seg_img_hands = yolo_segment_image(self.hand_seg_model, image)[0]
            if seg_path is not None and SAVE_SEG_IMAGES:
                seg_img_hands.save(seg_path + "-hands.png")
        # if seg_image_dir is None or not os.path.exists(seg_path):
        #     # process and segment the image
        #     orig_img, seg_img = yolo_segment_image(self.f_seg_model, image)[0]
        #     orig_img = orig_img.resize((image_resize_params.new_w, image_resize_params.new_h))
        #     if seg_image_dir is not None:
        #         seg_img.save(seg_path)
        # else:
        #     # load the segmentation from local disk instead
        #     orig_img = Image.fromarray(image).resize((image_resize_params.new_w, image_resize_params.new_h)).convert('RGB')
        #     seg_img = Image.open(seg_path).convert('RGB')

        seg_img = seg_img.resize((image_resize_params.new_w, image_resize_params.new_h))
        seg_img = np.array(seg_img, dtype=np.uint8)
        # if pad_mask is not None:
        #     seg_img[pad_mask == 255] = 255
        if face_masks_combined is not None:
            if self.settings.options[
                "keep_face"]:  # removes faces from seg; used to keep faces intact or for face restore
                # convert to numpy
                face_mask = face_masks_combined#cv2.resize(face_masks_combined, (image_resize_params.new_w, image_resize_params.new_h), interpolation=cv2.INTER_CUBIC)
                # seg_img = np.array(seg_img, dtype=np.uint8)
                seg_img[face_mask > 25] = 0  # set all pixels to black if face_mask(white fill) is above noise level

                if seg_path is not None and SAVE_SEG_IMAGES:
                    Image.fromarray(face_mask).save(seg_path + "-face_mask.png")

            if self.settings.options["keep_hands"]:  # removes hands from seg; used to keep hands intact
                # convert to numpy
                hand_mask = seg_img_hands.resize((image_resize_params.new_w, image_resize_params.new_h))
                hand_mask = np.array(hand_mask, dtype=np.uint8)

                seg_img[hand_mask > 127] = 0
            if pad_mask is not None:
                seg_img[pad_mask == 255] = 255
            seg_img = Image.fromarray(seg_img).convert('RGB')
            if seg_path is not None and SAVE_SEG_IMAGES:
                seg_img.save(seg_path + "-face.png")
        else:
            seg_img = Image.fromarray(seg_img).convert('RGB')
        # prepare to inpaint the segmented image
        control_image_inpaint = make_inpaint_condition(orig_img, seg_img, 0.5)

        return orig_img, seg_img, control_image_inpaint


def save_image(image, output_path):
    """
    Saves the given image to the output_path with a timestamped filename indicating the input type.

    :param image: Input image (NumPy array, PIL image, or Torch tensor)
    :param output_path: file path where the image will be saved
    :return: output_path:
    """
    if isinstance(image, np.ndarray):
        image_type = "numpy"
        image = Image.fromarray(image.astype(np.uint8))
    elif isinstance(image, Image.Image):
        image_type = "pil"
    elif isinstance(image, torch.Tensor):
        image_type = "tensor"
        if image.ndimension() == 3 and image.shape[0] in [1, 3]:  # CxHxW -> HxWxC
            image = image.permute(1, 2, 0)
        image = Image.fromarray((image.numpy() * 255).astype(np.uint8))
    else:
        raise TypeError("Unsupported image type. Must be NumPy array, PIL Image, or Torch Tensor.")

    # timestamp = time.strftime("%Y%m%d_%H%M%S")
    # filename = f"{timestamp}_{image_type}.png"
    # output_path = f"{output_dir}/{filename}"
    image.save(output_path)
    print(f"Image saved in {output_path}")
    return output_path


def load_image(image_path):
    """
    Loads an image from the given path and determines its original type based on filename.

    :param image_path: Path to the saved image file
    :return: Image in its original format (NumPy array, PIL image, or Torch tensor)
    """
    image = Image.open(image_path)
    filename = image_path.split("/")[-1]

    if "numpy" in filename:
        return np.array(image)
    elif "pil" in filename:
        return image
    elif "tensor" in filename:
        return torch.tensor(np.array(image) / 255.0).permute(2, 0, 1)
    else:
        raise ValueError("Could not determine image type from filename.")


if __name__ == "__main__":

    pipe_ids = ("flux", "flux-fill", "sd15", "sd15-fill")

    im_servers = [SocketServer(5000+i) for i in range(len(pipe_ids))]
    pipe_clients = [SocketClient(5100+i) for i in range(len(pipe_ids))]

    image_processor = ImageProcessor()

    while True:
        for pipe_idx, im_server in enumerate(im_servers):
            # get the inputs
            print(f"\r{int(time.time())}-[{pipe_idx}] waiting from port {im_server.port}...", end="")
            try:
                settings = im_server.get(blocking=False)
            except BlockingIOError:
                time.sleep(1)
                continue

            print(f"{pipe_ids[pipe_idx]}:{im_server.port} received settings:", settings)

            image_processor.set_seg_models(settings)

            # process a single image
            if os.path.isfile(settings.options['image']):
                # image mode
                outputs = image_processor.prepare_inputs(
                    settings.options['image'],
                    settings.options['mask'],
                    settings.options["max_side"],
                    settings.options["center_crop"])

                pipe_clients[pipe_idx].put(outputs)
            # process the directory
            else:
                image_dir = settings.options['image']
                for file in os.listdir(image_dir):
                    if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".webp") or file.endswith(".jfif"):
                        image_path = f"{image_dir}/{file}"
                        print(f"Processing {image_path}")
                        # try:
                        outputs = image_processor.prepare_inputs(
                            image_path,
                            None,
                            settings.options["max_side"],
                            settings.options["center_crop"])

                        # for every output,
                        pipe_clients[pipe_idx].put(outputs)

                        # except Exception as e:
                        #     print(e)