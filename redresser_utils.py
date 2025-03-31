import time
import socket
import pickle
import numpy as np
import cv2
from PIL import Image
import torch
import os

from diffusers import FluxTransformer2DModel, CogVideoXTransformer3DModel, GGUFQuantizationConfig, BitsAndBytesConfig
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import UMT5EncoderModel, T5EncoderModel

from tf_free_functions import align_crop_image, paste_swapped_image
from CONSTANTS import *


def pad_image(image, mask, pad_size, use_noise=False):
    '''

    :param image: (numpy image)
    :param mask: (numpy image)
    :param pad_size: (int | tuple) contains the amount to pad from left, top, right, bottom
    :param use_noise: (bool) use noise instead of a white bg and also return a padded mask
    :return: (numpy image) the padded image in
    '''
    h, w = image.shape[:2]
    # make sure the pad size is divisible by 2
    if isinstance(pad_size, int):
        corrected_pad_size = [int(pad_size if pad_size % 2 == 0 else pad_size + 1) for _ in range(4)]
    elif isinstance(pad_size, list):
        corrected_pad_size = [int(_pad_size if _pad_size % 2 == 0 else _pad_size + 1) for _pad_size in pad_size]
    elif isinstance(pad_size, str):
        pad_size = [int(i) for i in pad_size.split(",")]
        corrected_pad_size = [int(_pad_size if _pad_size % 2 == 0 else _pad_size + 1) for _pad_size in pad_size]
    else:
        raise ValueError(f"pad_size must be one of int or list, not {pad_size}")
    # pad_size = int(pad_size if pad_size % 2 == 0 else pad_size + 1)
    # half_pad_size = int(pad_size / 2)

    h_pad = corrected_pad_size[1] + corrected_pad_size[3]
    w_pad = corrected_pad_size[0] + corrected_pad_size[2]
    # pad = h_pad//2
    # # Apply reflective padding
    # padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    # return padded_image
    padded_white_background = np.zeros(shape=(h+h_pad, w+w_pad, image.shape[2]), dtype=np.uint8)
    padded_white_background += 255

    noise_background = np.random.randint(0, 256, (h+h_pad, w+w_pad, image.shape[2]), dtype=np.uint8)

    # Paste the image onto the noisy background
    noise_background[
        corrected_pad_size[1]: h + corrected_pad_size[1],
        corrected_pad_size[0]: w + corrected_pad_size[0]
    ] = image

    # Paste the image onto the white background
    padded_white_background[
        corrected_pad_size[1]: h + corrected_pad_size[1],
        corrected_pad_size[0]: w + corrected_pad_size[0]
    ] = image if not use_noise else np.zeros_like(image)

    if mask is not None:
        padded_mask_white_background = np.zeros(shape=(h + h_pad, w + w_pad, image.shape[2]), dtype=np.uint8)
        padded_mask_white_background += 255
        padded_mask_white_background[
            corrected_pad_size[1]: h + corrected_pad_size[1],
            corrected_pad_size[0]: w + corrected_pad_size[0]
        ] = mask
    else:
        padded_mask_white_background = None
    # padded_white_background[half_pad_size:half_pad_size + h, half_pad_size:half_pad_size + w] = image

    if use_noise:
        outputs = noise_background, padded_white_background, padded_mask_white_background
        return outputs
    else:
        return padded_white_background, padded_mask_white_background


def extract_xseg_mask(xseg_model, face_images, dilation=1.0):
    """Extracts the mask from the segmentation model. Resizes face_image to model input size and later resizes output
    back to original face_image dimensions
    Args:
        face_image (Numpy array): numpy image of a face

    Returns:
        (Numpy array): numpy image of a mask
    """

    inputs = []
    for face_image in face_images:

        # get the mask using image segmentation model
        if face_image.shape[0] * face_image.shape[1] > 256 * 256:
            face_image = cv2.resize(face_image, (256, 256), interpolation=cv2.INTER_AREA)
        elif face_image.shape[0] * face_image.shape[1] < 256 * 256:
            face_image = cv2.resize(face_image, (256, 256), interpolation=cv2.INTER_CUBIC)

        inputs.append(face_image)

    # minibatch
    swapped_faces = []
    mini_batch_size = 4
    mini_batch_faces = []
    for f_idx, face in enumerate(inputs):
        mini_batch_faces.append(face)
        if len(mini_batch_faces) == mini_batch_size or f_idx == len(inputs) - 1:
            _swapped_faces = np.array(xseg_model(np.array(mini_batch_faces)/255.))
            _swapped_faces = np.array(_swapped_faces, dtype=np.uint8)
            for swapped_face in _swapped_faces:
                swapped_faces.append(swapped_face)
            mini_batch_faces.clear()
    xseg_results = np.array(swapped_faces)
    # xseg_results = np.array(xseg_model(np.array(inputs)/255.))

    xseg_results[xseg_results < 0.1] = 0  # get rid of noise
    # xseg_results = np.clip(xseg_results, 0., 0.5) * 2  # binary
    xseg_results = np.clip(xseg_results * 255, 0, 255).astype('uint8')

    # print("seg_mask_after_squeeze", seg_mask.shape)
    seg_masks = []
    for seg_mask_index, seg_mask in enumerate(xseg_results):

        # np.clip(seg_mask * 255, 0, 255)

        # resize back to swapped image size if necessary; operations are also reversed
        # if face_images[seg_mask_index].shape[0] * face_images[seg_mask_index].shape[1] > 256 * 256:
        #     seg_mask = cv2.resize(seg_mask, (face_images[seg_mask_index].shape[0], face_images[seg_mask_index].shape[1]),
        #                           interpolation=cv2.INTER_CUBIC)
        # elif face_images[seg_mask_index].shape[0] * face_images[seg_mask_index].shape[1] < 256 * 256:
        #     seg_mask = cv2.resize(seg_mask, (face_images[seg_mask_index].shape[0], face_images[seg_mask_index].shape[1]),
        #                           interpolation=cv2.INTER_AREA)

        if dilation != 1.0:
            # transpose dilation
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            seg_mask = cv2.dilate(seg_mask, element)
            seg_mask = np.clip(seg_mask, 0, 255)
        else:
            # crop 3% of image and resize to expand mask area
            border_size = int(max(1, seg_mask.shape[0] * .03))

            h, w = seg_mask.shape[:2]
            # crop image
            seg_mask = seg_mask[border_size:h-border_size, border_size:w-border_size, :]
            seg_mask = cv2.resize(seg_mask, (h, w), interpolation=cv2.INTER_CUBIC)
            seg_mask = np.clip(seg_mask, 0, 255)

        # kernel = np.ones((kernel_size, kernel_size), np.uint8)
        #
        # # Perform dilation on the seg mask
        # seg_mask = cv2.dilate(seg_mask, kernel, iterations=1)
        seg_masks.append(seg_mask)

    return seg_masks


class RedresserSettings:
    SEGMENT_FASHION = 0
    SEGMENT_PERSON = 1
    SEGMENT_FACE = 2  # Ignores keep_face; FaceGen
    SEGMENT_ALL = 3  # replaces whole image unless keep_face or keep_hands is True

    # why are these here? FOR CONVENIENCE
    xseg_model_path = "xseg/saved_model"  # local path only
    face_model_path = "yolov8n-face.pt"  # local path only
    face_box_model_path = "face_yolov9c.pt"  # downloads from Bingsu/adetailer

    person_model_path = "person_yolov8m-seg.pt"  # downloads from Bingsu/adetailer
    fashion_model_path = "deepfashion2_yolov8s-seg.pt"  # downloads from Bingsu/adetailer
    hand_model_path = "hand_yolov9c.pt"  # downloads from Bingsu/adetailer
    default_options = {
        # "prompt": "best quality 4k, <jjk-d-step00000300>close-up shot of a jjk face, shoulder-length hair",
        "prompt": "",
        "image": "dance",
        # "mask": "seg/00000.png",
        "guidance_scale": 3.5,
        "num_inference_steps": 8,
        # "negative_prompt": None,
        # "strength": None,
        # "clip_skip": 0,
        "seed": -1,
        "max_side": 1024,  # used for fill
        "height": 1024,  # used for t2i only
        "width": 1024,  # used for t2i only
        "center_crop": False,
        "padding": 0,
        "SEGMENT_ID": SEGMENT_PERSON,
        "keep_hands": True,
        "keep_face": True,
        # "face_mask_scale": 1.0,
        # "use_faceswap": False,
        "runs": 1,  # how many times to run with these settings
    }

    def __init__(self, ):
        self.options = {}
        self.previous_options = self.__class__.default_options.copy()

    def map_dfs_options(self, dfs_options, model):

        self.options["prompt"] = dfs_options.get("prompt", "")

        if model == "fill":
            print("setting guidance_scale to 10x for repainter")
            self.options["guidance_scale"] = dfs_options.get("cfg", 3.0) * 10
            self.options["num_inference_steps"] = dfs_options.get("steps", 8)

            self.options["max_side"] = dfs_options.get("size", 1024)
            self.options["padding"] = dfs_options.get("padding", 0)
            self.options["center_crop"] = dfs_options.get("center_crop", False)
            self.options["SEGMENT_ID"] = RedresserSettings.SEGMENT_PERSON  # dfs_options.get("autoMaskOption", RedresserSettings.SEGMENT_PERSON)

            if dfs_options.get("maskOption") == 0:
                self.options["keep_hands"] = dfs_options.get("preserveHands", True)
                self.options["keep_face"] = dfs_options.get("preserveHead", True)
            else:
                self.options["keep_hands"] = False
                self.options["keep_face"] = False

            if self.options["max_side"] == 0:
                self.options["max_side"] = 1024
        else:
            cfg = dfs_options.get("cfg", 3.5)
            if 1.0 > cfg or 10 < cfg:
                print("setting guidance_scale to 3.5 for turbo")
                cfg = 3.5

            self.options["guidance_scale"] = cfg
            self.options["num_inference_steps"] = dfs_options.get("steps", 8)

            self.options["height"] = dfs_options.get("height", 1024)
            self.options["width"] = dfs_options.get("width", 1024)

            if self.options["height"] == 0:
                self.options["height"] = 1024
            if self.options["width"] == 0:
                self.options["width"] = 1024

        self.options["runs"] = dfs_options.get("runs", 1)

        # don't check mask and image (if it doesn't exist, it'll use default value, and we don't want that)

        try:
            self.options.pop("mask")
        except KeyError:
            pass

        try:
            self.options.pop("image")
        except KeyError:
            pass

        self.check_inputs()

        # add the image and mask back here
        self.options["mask"] = dfs_options.get("mask", None)
        self.options["image"] = dfs_options.get("image", None)

        self.previous_options = self.options.copy()

    def set_options(self):
        use_previous_for_rest = False
        for key, val in self.__class__.default_options.items():
            if use_previous_for_rest:
                self.options[key] = self.previous_options[key]
                print(f"{key}: {self.options[key]} (using previous config)")
                continue

            self.options[key] = input(f'{key}({self.previous_options[key]}):')
            print("You've entered:", self.options[key])

            # PREVIOUS KEY
            if self.options[key] == '':
                self.options[key] = self.previous_options[key]
            # ALL PREVIOUS KEY
            elif self.options[key] == 'p':
                use_previous_for_rest = True
                self.options[key] = self.previous_options[key]
                print(f"{key}: {self.options[key]} (using previous config)")
            # RESET KEY
            elif self.options[key] == 'r':
                self.options[key] = val

        self.check_inputs()

        self.previous_options = self.options.copy()

        if use_previous_for_rest:
            print("using previous config for the rest:", self.previous_options)

    def check_inputs(self, ):
        for key, val in self.options.items():
            if key in ['guidance_scale', 'face_mask_scale', 'padding']:
                try:
                    self.options[key] = float(self.options[key])
                except:
                    self.options[key] = self.__class__.default_options[key]
            if key in ['max_area', 'num_frames', 'num_inference_steps', 'max_side', 'height', 'width', "strength", "clip_skip", "seed", "SEGMENT_ID"]:
                try:
                    self.options[key] = int(self.options[key])
                except:
                    self.options[key] = self.__class__.default_options[key]
            if key in ['use_dynamic_cfg', 'center_crop', "keep_hands", 'keep_face', 'use_faceswap']:
                try:
                    bool_opt = self.options[key]
                    if isinstance(bool_opt, str):
                        if bool_opt.lower() == 'true':
                            self.options[key] = True
                        elif bool_opt.lower() == 'false':
                            self.options[key] = False
                        else:
                            self.options[key] = bool(self.options[key])
                    elif isinstance(bool_opt, bool):
                        self.options[key] = bool_opt

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


# returns a list of pairs containing the input image and the output mask


def make_inpaint_condition(image, image_mask, mask_threshold=0.5):
    '''
    prepares the inpaint image for SD pipeline

    image(PIL.Image): input image
    image_mask(PIL.Image): input mask
    mask_threshold(float): any pixel values above this threshold is set to -1.0
    return(Torch.tensor): the tensor to pass to SD
    '''
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1]
    image[image_mask >= mask_threshold] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def yolo8_extract_faces(face_extractor, face_seg_model, inputs, max_faces, conf_threshold, landmarks_padding_ratio, include_orig_img=False, face_swapper_input_size=(224, 224)):
    _outputs = face_extractor(inputs, tracker=False)  # frames in rgb, convert to PIL if necessary
    post_processing_secs = time.time()

    batch_extracted_new_params = {}
    outputs = []
    for output in _outputs:
        outputs.append(output.cpu().numpy())

    for out_idx, output in enumerate(outputs):
        orig_img = output.orig_img
        orig_img_bw = np.zeros_like(orig_img)[:, :, 0].astype(np.uint8)
        orig_img_bw = np.expand_dims(orig_img_bw, -1)
        orig_img_bw = np.concatenate((orig_img_bw, orig_img_bw, orig_img_bw), -1)
        batch_extracted_new_params[out_idx] = {}

        if len(output) == 0:
            batch_extracted_new_params[out_idx][0] = {
                "orig_img": orig_img.copy() if include_orig_img else None,
                "seg_mask_combined": None
            }
            continue

        if len(output) > 0:
            keypoints = output.keypoints.xy
            boxes_data = output.boxes.data
            boxes_xyxy = boxes_data[:, :4]
            confs = boxes_data[:, -2]

            for idx, xy in enumerate(keypoints):  # tensor of (ids,5,2)
                conf = confs[idx]
                # print("xy", xy, "conf", conf)  # individual face keypoints of shape(5, 2)
                if conf < 0.45:
                    continue

                if idx+1 > max_faces:
                    break

                # align and crop the face
                aligned_cropped_image, aligned_cropped_params = align_crop_image(
                    original_image=orig_img,
                    original_landmarks=xy,
                    landmarks_padding_ratio=landmarks_padding_ratio
                )

                # return if face is invalid
                if aligned_cropped_image.shape[0] >= orig_img.shape[0] or aligned_cropped_image.shape[1] >= orig_img.shape[1]:
                    continue
                cv2.imwrite(f"seg/seg-aligned-{idx}.png", aligned_cropped_image)
                # resize cropped image to face swap input size
                try:
                    processed_image = cv2.resize(aligned_cropped_image, face_swapper_input_size, interpolation=cv2.INTER_CUBIC)
                except cv2.error:
                    continue

                if face_seg_model is None:
                    seg_mask = (np.ones_like(processed_image) * 255).astype("uint8")
                else:
                    seg_mask = extract_xseg_mask(xseg_model=face_seg_model, face_images=[processed_image])[0]

                cv2.imwrite(f"seg/seg-mask-{idx}.png", seg_mask)

                batch_extracted_new_params[out_idx][idx] = {
                    "aligned_cropped_image": aligned_cropped_image,
                    "aligned_cropped_params": aligned_cropped_params,
                    "processed_image": processed_image,  # needs to be resized(224x224), rescaled(0,1) image
                    "bbox": boxes_xyxy[idx],
                    "seg_mask": seg_mask,  # 256x256 output of face_seg_model
                    # "orig_img": orig_img.copy() if include_orig_img else None
                }

            # combine face masks
            if len(batch_extracted_new_params[out_idx]) > 0:
                for face_idx, face_data in batch_extracted_new_params[out_idx].items():
                    aligned_cropped_image = face_data["aligned_cropped_image"]
                    seg_mask = face_data["seg_mask"]
                    seg_mask = cv2.resize(seg_mask, (aligned_cropped_image.shape[1], aligned_cropped_image.shape[0]), cv2.INTER_CUBIC)
                    # seg_mask = np.expand_dims(seg_mask, -1)
                    # seg_mask = np.concatenate((seg_mask, seg_mask, seg_mask), -1)
                    orig_img_bw = paste_swapped_image(
                        dst_image=orig_img_bw,
                        swapped_image=seg_mask,
                        seg_mask=seg_mask,
                        aligned_cropped_params=face_data["aligned_cropped_params"],
                        seamless_clone=False,
                        blur_mask=True,
                        resize=False
                    )

                    # xyxy = face_data["bbox"]
                    # seg_mask = face_data["seg_mask"]
                    # h = int(xyxy[3] - xyxy[1])
                    # w = int(xyxy[2] - xyxy[0])
                    # seg_mask = cv2.resize(seg_mask, (w, h), cv2.INTER_CUBIC)
                    # orig_img_bw[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] = seg_mask
                    cv2.imwrite(f"seg/seg-combined-{face_idx}.png", orig_img_bw)

                for face_idx, face_data in batch_extracted_new_params[out_idx].items():
                    face_data["orig_img"] = orig_img.copy() if include_orig_img else None
                    face_data["seg_mask_combined"] = orig_img_bw[:, :, 0]  # just get first channel
            else:
                batch_extracted_new_params[out_idx][0] = {
                    "orig_img": orig_img.copy() if include_orig_img else None,
                    "seg_mask_combined": orig_img_bw[:, :, 0]  # just get first channel
                }


    print(f"postprocess face extractor outputs finished in",
          time.time() - post_processing_secs)

    return batch_extracted_new_params


def yolo_segment_image(yolo_model, img):
        outputs = yolo_model(img, tracker=False)
        seg_results = []
        for output in outputs:
            imgg = output.orig_img
            # imgg = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)

            # debug
            # pred = output.plot()
            # pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            # pred = Image.fromarray(pred)
            # pred.show()

            empty_img = np.zeros_like(imgg)
            if output.masks is not None:
                # mask = mask.astype('uint8')  # Convert to int32 for use with cv2 functions
                # cv2.fillConvexPoly(imgg, [mask], color=(0, 0, 0))  # Black mask
                for mask_idx, mask in enumerate(output.masks.xy):
                    conf = output.boxes.conf[mask_idx]
                    print("mask", mask.shape, "conf", conf)
                    if conf > 0.2:
                        mask = mask.astype(np.int32)  # Convert to int32 for use with cv2 functions
                        cv2.fillPoly(empty_img, [mask], color=(255, 255, 255))  # white mask over black
            elif output.boxes is not None:
                boxes = output.boxes.xyxy.cpu().numpy()  # Convert tensor to NumPy array if needed

                # Loop over each detected box and draw a white rectangle on the mask
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)  # Convert to integers for drawing
                    cv2.rectangle(empty_img, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=-1)

            seg_results.append([Image.fromarray(imgg), Image.fromarray(empty_img)])

        return seg_results


def load_image(image_path, return_type="pil"):
    """
    Loads an image from the given path and determines its original type based on filename.

    :param image_path: Path to the saved image file
    :param return_type: PIL numpy or Torch tensor to return
    :return: Image in its original format (NumPy array, PIL image, or Torch tensor)
    """
    image = Image.open(image_path)
    filename = image_path.split("/")[-1]

    if "numpy" in return_type:
        return np.array(image)
    elif "pil" in return_type:
        return image
    elif "tensor" in return_type:
        return torch.tensor(np.array(image) / 255.0).permute(2, 0, 1)


class SocketServer:
    def __init__(self, port=5000):
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(("localhost", self.port))
        self.server.listen(5)

    def get(self, blocking=True, use_pickle=True):
        self.server.setblocking(blocking)

        client, _ = self.server.accept()

        data = client.recv(4096*2)  # Receive serialized object
        print(f"{self.__class__.__name__} {self.port} received object")
        if use_pickle:
            obj = pickle.loads(data)  # Deserialize
        else:
            obj = data

        client.close()
        return obj


class SocketClient:
    def __init__(self, port=5000):
        self.port = port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def put(self, data, max_retries=99, retry_wait=1, use_pickle=True):
        cur_retries = 0
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.client.connect(("localhost", self.port))
                break
            except ConnectionRefusedError:
                cur_retries += 1
                if cur_retries > max_retries:
                    return
                print(f"Put failed! Retrying in {retry_wait} seconds")
                time.sleep(retry_wait)
        if use_pickle:
            data = pickle.dumps(data)
        print(f"{self.__class__.__name__} {self.port} sending object")
        self.client.send(data)
        self.client.close()


def push_model_to_hub():
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    # cog_model_id = "D:/huggingface/models--THUDM--CogVideoX-5b-I2V/snapshots/c5c783ca1606069b9996dc56f207cc2e681691ed"
    #
    quant_config = DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    # model = CogVideoXTransformer3DModel.from_pretrained(
    #     cog_model_id,
    #     subfolder="transformer",
    #     # 'https://huggingface.co/Kijai/CogVideoX_GGUF/resolve/main/CogVideoX_5b_1_5_I2V_GGUF_Q4_0.safetensors',
    #     quantization_config=quant_config,
    # )
    # print("saving model")
    # model.save_pretrained("cogvideox-nf4", max_shard_size=SHARD_SIZE)
    # model = FluxTransformer2DModel.from_pretrained(
    #             FLUX_CUSTOM_PATH,
    #             # subfolder="transformer",
    #             local_files_only=USE_LOCAL_FILES
    #         )
    # print("pushing to hub")
    # model.push_to_hub("cogvideox-nf4")
    print("loading model")
    model = FluxTransformer2DModel.from_pretrained(
        FLUX_FILL_PATH,
        # "C:\\Users\\teckt\\.cache\\huggingface\\hub\\models--Kijai--flux-fp8\\snapshots\\e77f550e3fe8be226884d5944a40abdbe4735ff5\\flux1-dev-fp8.safetensors",
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16, local_files_only=True)

    # model.to("cuda")
    print("saving model")
    model.save_pretrained("flux-fill-nf4", max_shard_size="16GB")
    # with tqdm(range(4), desc="Fusing hyper lora") as p:
    #     model.load_lora_adapter(
    #         hf_hub_download(FUSE_HYPER_LORA_REPO, FUSE_HYPER_LORA_MODEL_FILE),
    #         adapter_name="hyper")
    #     p.update()
    #     p.desc = "Setting adapter"
    #     lora_settings = {"adapter_names": ["hyper"],
    #                      "weights": [.125]}
    #     model.set_adapters(**lora_settings)
    #     p.update()
    #     p.desc = "Fusing lora"
    #     model.fuse_lora()
    #     p.update()
    #     p.desc = "Unloading lora"
    #     model.unload_lora()
    #     p.desc = "Deleting adapter"
    #     model.delete_adapters(["hyper"])
    #     p.update()

    # print("pushing to hub")
    # print("converting to f8")
    # model.to(torch.float8_e4m3fn)
    # print("saving")
    # model.save_pretrained("flux-hyper-fp8", max_shard_size="32GB")
    # model.push_to_hub("flux-fill-nf4")
    # del model
    # quant_config = BitsAndBytesConfig(load_in_8bit=True)
    # print("loading into bnb")
    # model = FluxTransformer2DModel.from_pretrained(
    #     "flux-hyper-fp16",
    #     quantization_config=quant_config,
    #     torch_dtype=torch.bfloat16
    # )
    # print("saving!!")
    # model.save_pretrained("flux-hyper-q8", max_shard_size="32GB")


if __name__ == "__main__":
    push_model_to_hub()