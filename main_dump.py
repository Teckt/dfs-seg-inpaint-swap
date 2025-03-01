from safetensors.torch import load_file

from dfs_quick_swap.main import extract_xseg_mask
from models.xseg.iae_xseg import build_seg_model

xseg_model_path = "C:/Users/teckt/xseg/models/saved_model"

import os
import time

import numpy as np
from controlnet_aux.processor import Processor
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor2_0
from diffusers.utils.export_utils import _legacy_export_to_video
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

import cv2
from PIL import Image

import torch
from diffusers import EulerAncestralDiscreteScheduler, ControlNetModel, \
    DEISMultistepScheduler, DDIMScheduler, LCMScheduler, \
    AutoencoderKL, StableDiffusionXLControlNetPipeline, StableDiffusionXLControlNetInpaintPipeline, \
    DPMSolverMultistepScheduler, AnimateDiffSDXLPipeline, AnimateDiffSparseControlNetPipeline, \
    StableDiffusionControlNetPipeline, StableDiffusionControlNetInpaintPipeline, StableDiffusionControlNetPipeline, \
    StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline, MotionAdapter, AnimateDiffControlNetPipeline, \
    SparseControlNetModel

from dfs_quick_swap.tf_free_functions import align_crop_image, paste_swapped_image
# from stable_diffusion.custom_pipeline import StableDiffusionControlNetInpaintPipeline
from stable_diffusion.animatediffvideo2videocontrolnetpipeline import AnimateDiffVideoToVideoControlNetPipeline

from stable_diffusion.helpers import load_textual_inversions, load_image_loras, ImgToImgOpticalFlow, export_to_gif, \
    load_image_loras_xl, USE_ANIMATE_LCM
from yolomodel.process_opt_flow import run_ffmpeg_optical_flow



class ImageResizeParams:
    '''
    -init
    h(int): desired image output height in pixels
    w(int): desired image output width in pixels
    max_side(int): maximum size of any side in pixels
    center_crop(bool): True to center crop a square and use max(h,w) as the size in pixels, False to resize and keep aspect ratio
    '''
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
        '''
        image(PIL.Image, numpy.ndarray): image to apply the init args 
        return(PIL.Image, numpy.ndarray): the resized image in the same format
        '''
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

def make_inpaint_condition(image, image_mask, mask_threshold=0.5):
    '''
    prepares the inpaint image for SD pipeline

    image(PIL.Image): input image
    image_mask(PIL.Image): input mask
    mask_threshold(float): any pixel values above this threshold is set to -1.0 
    return(Torch.tensor): the readied tensor to pass to SD
    '''
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1]
    image[image_mask >= mask_threshold] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

# def make_inpaint_condition(image, image_mask):
#     image = np.array(image.convert("RGB")).astype(np.uint8)
#     image_mask = np.array(image_mask.convert("L")).astype(np.uint8)
#
#     assert image.shape[0:1] == image_mask.shape[0:1]
#     image[image_mask > 127] = -255  # set as masked pixel
#     image = Image.fromarray(image)
#     # image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
#     # image = torch.from_numpy(image)
#     return image

def mask_control_condition(control_image, image_mask):
    return control_image
    image = np.array(control_image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1]
    image[image_mask < 0.5] = 0  # set as masked pixel

    return Image.fromarray((image*255).astype(np.uint8))
    # image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    # image = torch.from_numpy(image)
    # return image


class SegmentInpainter:
    def __init__(self, control_p, use_animate_diff, use_animate_diff_v2v, free_noise_params, free_noise_split_params, use_lightning, use_sparsectrl, use_xl, use_inpainting, use_multiple_controls, segment_type):

        if segment_type not in ['person', 'fashion']:
            print("segment_type does not exist:", segment_type)
            assert False

        self.xseg_model = build_seg_model(resolution=(256, 256), load=True, save_path=xseg_model_path)
        self.face_extract_model = YOLO("yolov8n-face.pt")

        self.use_lightning = use_lightning
        self.use_animate_diff_v2v = use_animate_diff_v2v
        self.free_noise_params = free_noise_params
        self.free_noise_split_params = free_noise_split_params
        self.use_sparsectrl = use_sparsectrl
        self.use_inpainting = use_inpainting
        self.use_multiple_controls = use_multiple_controls
        self.segment_type = segment_type
        # inpaint pipeline with 2 controls (image, depth)
        # self.canny_p = Processor("canny")
        # self.depth_p = Processor("depth_midas")
        if control_p in ['depth_midas', 'canny', 'openpose', 'normal_bae']:
            self.control_p = Processor(control_p)
        else:
            self.control_p = None
        if use_xl and self.control_p is None:
            # must have a control that's not inpainting
            assert False

        self.use_zero = True
        self.use_animate_diff = use_animate_diff
        self.use_xl = use_xl
        self.pipeline = self.get_pipeline()

        if self.segment_type == 'person':
        # segments images and fill with black
            seg_item = "person_yolov8m-seg.pt"
        if self.segment_type == 'fashion':
            seg_item = "deepfashion2_yolov8s-seg.pt"
        # if self.segment_type == 'face':
        #     seg_item = "yolov8n-face.pt"
        #     self.f_seg_model = YOLO(seg_item)
        # else:
        # # seg_item = "hand_yolov9c.pt"

        self.f_seg_model = YOLO(hf_hub_download("Bingsu/adetailer", seg_item))

    def get_pipeline(self):
        if self.use_animate_diff and self.use_xl:
            print("animatediff not implemented with xl")
            assert False

        # model_id = "benjamin-paine/stable-diffusion-v1-5-inpainting"
        if self.use_xl:
            model_id = "SG161222/RealVisXL_V5.0"

            # model_id = "RunDiffusion/Juggernaut-XI-v11"
        else:
            model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
            if self.use_inpainting and not self.use_animate_diff:
                # can't do anything other than 512x512
                model_id = "C:\\Users\\teckt\\hf\\models\\realisticVisionV60B1_v60B1InpaintingVAE.safetensors"

        if self.use_xl:
            pipeline_args = {
                "pretrained_model_name_or_path": model_id,
                "safety_checker": None,
                "torch_dtype": torch.float16,
                "use_safetensors": True,
                "variant": "fp16",
            }

            if self.control_p.processor_id == 'depth_midas':
                control_net_id = "xinsir/controlnet-depth-sdxl-1.0"
            if self.control_p.processor_id == 'canny':
                control_net_id = "diffusers/controlnet-canny-sdxl-1.0"
            if self.control_p.processor_id == 'openpose':
                control_net_id = "xinsir/controlnet-openpose-sdxl-1.0"

            controlnet = ControlNetModel.from_pretrained(
                control_net_id, torch_dtype=torch.float16
            )
            controlnet.set_attn_processor(CrossFrameAttnProcessor2_0(batch_size=2))

            if self.use_animate_diff:
                controlnet = ControlNetModel.from_pretrained(
                    'destitech/controlnet-inpaint-dreamer-sdxl', torch_dtype=torch.float16, variant="fp16",
                )
            pipeline_args["controlnet"] = controlnet
            if self.use_animate_diff:
                pipeline = AnimateDiffSDXLPipeline.from_pretrained(**pipeline_args)
            else:
                if self.use_inpainting:

                    # _pipeline = StableDiffusionXLPipeline.from_pretrained(**pipeline_args)
                    # change for inpainting
                    # pipeline_args.pop("pretrained_model_name_or_path")
                    # pipeline_args.pop("variant")
                    # pipeline_args.pop("use_safetensors")

                    pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(**pipeline_args)

                    # pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(**pipeline_args)
                else:
                    pipeline_args["controlnet"] = controlnet
                    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(**pipeline_args)

            # vae = AutoencoderKL.from_pretrained(
            #     "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
            # )
            # pipeline_args["vae"] = vae

            # pipeline = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
            #     "https://huggingface.co/RunDiffusion/Juggernaut-XI-v11/blob/main/Juggernaut-XI-byRunDiffusion.safetensors",
            #     safety_checker=None,
            #     torch_dtype=torch.float16,
            # )

            # Set the attention processor
            pipeline.unet.set_attn_processor(CrossFrameAttnProcessor2_0(batch_size=2))

            # enable memory savings
            pipeline.enable_model_cpu_offload()
            pipeline.enable_vae_slicing()
            pipeline.enable_vae_tiling()

            # load_textual_inversions(pipeline)
            # load_image_loras_xl(pipeline)
        else:

            print("loading controlnets")

            if self.control_p is not None:
                if self.control_p.processor_id == 'depth_midas':
                    control_net_id = "lllyasviel/control_v11f1p_sd15_depth"
                if self.control_p.processor_id == 'canny':
                    control_net_id = "lllyasviel/sd-controlnet-canny"
                if self.control_p.processor_id == 'openpose':
                    control_net_id = 'lllyasviel/control_v11p_sd15_openpose'
                if self.control_p.processor_id == 'normal_bae':
                    control_net_id = 'lllyasviel/control_v11p_sd15_normalbae'

                controlnet = ControlNetModel.from_pretrained(control_net_id, torch_dtype=torch.float16, local_files_only=True)
                # controlnet.set_attention_slice("max")

            if not self.use_animate_diff:
                controlnet.set_attn_processor(CrossFrameAttnProcessor2_0(batch_size=2))
            else:
                if self.use_inpainting:
                    if self.use_multiple_controls:
                        inpaint_controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, local_files_only=True)
                        # inpaint_controlnet.set_attention_slice("max")
                        controlnet = [
                            inpaint_controlnet,
                            controlnet,
                        ]
                    else:
                        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint",
                                                                     torch_dtype=torch.float16, local_files_only=True)
                        # controlnet.set_attention_slice("max")

            print("using StableDiffusionControlNetInpaintPipeline")
            if self.use_animate_diff:
                if USE_ANIMATE_LCM:
                    motion_id = "wangfuyun/AnimateLCM"
                    adapter = MotionAdapter.from_pretrained(motion_id, local_files_only=True, torch_dtype=torch.float16)
                else:
                    if self.use_lightning:
                        device = "cuda"
                        dtype = torch.float16
                        step = 2  # Options: [1,2,4,8]
                        repo = "ByteDance/AnimateDiff-Lightning"
                        ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
                        adapter = MotionAdapter().to(device, dtype)
                        adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))

                    else:
                        # motion_id = "guoyww/animatediff-motion-adapter-v1-5-3"
                        motion_id = "https://huggingface.co/Lightricks/LongAnimateDiff/blob/main/lt_long_mm_32_frames.ckpt"
                        adapter = MotionAdapter.from_single_file(motion_id, torch_dtype=torch.float16, local_files_only=True)

                if self.use_sparsectrl:
                    controlnet = SparseControlNetModel.from_pretrained("guoyww/animatediff-sparsectrl-rgb",
                                                                 torch_dtype=torch.float16, local_files_only=True)
                    motion_id = "guoyww/animatediff-motion-adapter-v1-5-3"
                    adapter = MotionAdapter.from_pretrained(motion_id, torch_dtype=torch.float16, local_files_only=True)
                    pipeline = AnimateDiffSparseControlNetPipeline.from_pretrained(
                        model_id,
                        motion_adapter=adapter,
                        controlnet=controlnet,
                        torch_dtype=torch.float16,
                        local_files_only=True
                    )
                else:
                    if self.use_animate_diff_v2v:
                        pipeline = AnimateDiffVideoToVideoControlNetPipeline.from_pretrained(
                            model_id,
                            motion_adapter=adapter,
                            controlnet=controlnet,
                            torch_dtype=torch.float16,
                            safety_checker=None,
                            local_files_only=True
                        )

                    else:
                        pipeline = AnimateDiffControlNetPipeline.from_pretrained(
                            model_id,
                            motion_adapter=adapter,
                            controlnet=controlnet,
                            torch_dtype=torch.float16,
                            safety_checker=None,
                            local_files_only=True
                        )
                        # Enable FreeNoise for long prompt generation
                        # pipeline.enable_free_noise(context_length=self.context_length, context_stride=self.context_stride)


            else:
                if self.use_inpainting:
                    pipeline = StableDiffusionInpaintPipeline.from_single_file(
                        model_id, torch_dtype=torch.float16, use_safetensors=True,
                        local_files_only=True
                    )
                    # pipeline = StableDiffusionControlNetInpaintPipeline.from_single_file(
                    #     model_id, torch_dtype=torch.float16, controlnet=controlnet,
                    #     use_safetensors=True
                    # )
                else:
                    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                        model_id, torch_dtype=torch.float16, controlnet=controlnet,
                        use_safetensors=True, local_files_only=True
                    )
                pipeline.unet.set_attn_processor(CrossFrameAttnProcessor2_0(batch_size=2))

            if 'novae' in model_id.lower():
                vae = AutoencoderKL.from_pretrained(
                    "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16, local_files_only=True
                )
                pipeline.vae = vae

            # pipeline.to('cuda')
            load_textual_inversions(pipeline)
            load_image_loras(pipeline)
            pipeline.fuse_lora()

            # if self.use_animate_diff_v2v:
                # Enable FreeNoise for long prompt generation
            if self.free_noise_params:
                pipeline.enable_free_noise(context_length=self.free_noise_params[0], context_stride=self.free_noise_params[1])
            if self.free_noise_split_params:
                pipeline.enable_free_noise_split_inference(spatial_split_size=self.free_noise_split_params[0], temporal_split_size=self.free_noise_split_params[1])

                # pipeline.unet.enable_attn_chunking(
                #     self.free_noise_params[0])  # Temporal chunking across batch_size x num_frames
                # pipeline.unet.enable_motion_module_chunking(
                #     (512 // 8 // 4) ** 2
                # )  # Spatial chunking across batch_size x latent height x latent width
                # pipeline.unet.enable_resnet_chunking(self.free_noise_params[0])
                # pipeline.unet.enable_forward_chunking(self.free_noise_params[0])

            pipeline.enable_model_cpu_offload()
            # pipeline.enable_sequential_cpu_offload()
            pipeline.enable_vae_slicing()
            pipeline.enable_vae_tiling()

        return pipeline

    def set_scheduler(self, scheduler):
        self.pipeline.scheduler = scheduler

    # returns a list of pairs containing the input image and the output mask
    def segment_image(self, img):
        outputs = self.f_seg_model(img, tracker=False)
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
            seg_results.append([Image.fromarray(imgg), Image.fromarray(empty_img)])

        return seg_results

    def process_control_video(self, control_filename_no_etx, control_input_path, start_frame, frame_skip, image_resize_params, batch_size):
        print(f"processing control batch({batch_size}) at frame={start_frame} for {control_input_path}")
        # cache the control images for each new video's id and control processor
        if self.control_p is None:
            assert False

        if not os.path.exists(control_filename_no_etx):
            os.mkdir(control_filename_no_etx)
        control_image_dir = f"{control_filename_no_etx}\\{self.control_p.processor_id}_{'cc' if image_resize_params.center_crop else ''}"
        if not os.path.exists(control_image_dir):
            os.mkdir(control_image_dir)

        control_frames = []
        frame_index = start_frame

        current_index = 0

        cap = cv2.VideoCapture(control_input_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            if frame_skip == 1 or current_index % frame_skip == 0:
                control_image_path = f"{control_image_dir}\\{str(frame_index).zfill(5)}.png"
                if os.path.exists(control_image_path):
                    control_image_p = Image.open(control_image_path)
                else:
                    # outputs same type (only accepts PIL Image or nparray
                    frame = image_resize_params.apply_params(frame)
                    control_image_p = self.control_p(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    # output is a PIL Image
                    control_image_p.save(control_image_path)

                # convert to RGB
                control_image_p = control_image_p.resize((image_resize_params.new_w, image_resize_params.new_h)).convert('RGB')
                control_frames.append(control_image_p)
                # control_image_p.show("control")
            if len(control_frames) == batch_size:
                break
            frame_index += 1
            current_index += 1
        print(f"returning {len(control_frames)} control_frames")
        return control_frames

    def set_initial_frames(self):
        if is_image:
            initial_video_control_frame = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)

        if not is_image:
            output_video = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mpg2'),
                fps / frame_skip,  # fps
                (new_w, new_h)
            )
            # get the init frame and reset index
            cap.set(cv2.CAP_PROP_POS_FRAMES, initial_video_control_index)
            ret, initial_video_control_frame = cap.read()
            if not ret:
                raise ValueError(f"frame index for initial_video_control_frame{initial_video_control_frame} doesn't exist!")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # resize initial images
        if self.attach_initial_frame:
            if center_crop:
                initial_video_control_frame = initial_video_control_frame[ymin:ymax, xmin:xmax, :]
                initial_video_control_frame = cv2.resize(initial_video_control_frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            else:
                initial_video_control_frame = cv2.resize(initial_video_control_frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            # segment the initial image
            self.initial_orig_img, self.initial_seg_img = self.segment_image(initial_video_control_frame)[0]
            self.initial_orig_img = self.initial_orig_img.resize((new_w, new_h))
            self.initial_seg_img = self.initial_seg_img.resize((new_w, new_h))

            self.initial_control_img_path = f"{control_image_dir}\\{str(initial_video_control_index).zfill(5)}.png"
            if os.path.exists(self.initial_control_img_path):
                self.initial_control_img = Image.open(self.initial_control_img_path).convert('RGB')
            else:
                self.initial_control_img = self.control_p(self.initial_orig_img)
                self.initial_control_img.save(self.initial_control_img_path)

            self.initial_control_img = self.initial_control_img.resize((new_w, new_h))
            self.initial_control_img = mask_control_condition(self.initial_control_img, self.initial_seg_img)

    def pre_process_image_for_input(self, image, video_index, image_resize_params, seg_image_dir, seg_mask_combined, use_only_face_mask, use_negative_face_mask):
        # image should be in RGB
        if not os.path.exists(seg_image_dir):
            os.mkdir(seg_image_dir)

        seg_path = f"{seg_image_dir}\\{str(video_index).zfill(5)}.png"

        if not os.path.exists(seg_path):
            orig_img, seg_img = self.segment_image(image)[0]
            orig_img = orig_img.resize((image_resize_params.new_w, image_resize_params.new_h))
            seg_img.save(seg_path)
        else:
            orig_img = Image.fromarray(image).resize((image_resize_params.new_w, image_resize_params.new_h)).convert('RGB')
            seg_img = Image.open(seg_path).convert('RGB')

        seg_img = seg_img.resize((image_resize_params.new_w, image_resize_params.new_h))
        if seg_mask_combined is not None:
            if use_only_face_mask:  # replaces seg with only faces
                face_mask = seg_mask_combined
                seg_img = Image.fromarray(face_mask).convert('RGB')
            elif use_negative_face_mask:  # removes faces from seg; used to keep faces intact or for face restore
                face_mask = seg_mask_combined
                seg_img = np.array(seg_img, dtype=np.uint8)
                seg_img[face_mask > 127] = 0
                seg_img = Image.fromarray(seg_img).convert('RGB')

        # prepare to inpaint the segmented image
        control_image_inpaint = make_inpaint_condition(orig_img, seg_img, 0.5)

        return orig_img, seg_img, control_image_inpaint

    def process_image_for_video(self, input_path, filename_no_etx, image_resize_params, batch_size, use_only_face_mask, use_negative_face_mask):
        print(f"processing image batch({batch_size}) at {input_path}")
        frame = Image.open(input_path).convert('RGB')

        # frame = cv2.imread(input_path)
        frame = image_resize_params.apply_params(frame)
        batch_frames = {}
        orig_imgs = []
        seg_imgs = []
        control_images_inpaint = []
        for i in range(batch_size):
            batch_frames[i] = frame

        # for face segmentation
        yolo_results = extract_predict_yolo8(
            face_extractor=self.face_extract_model, face_seg_model=self.xseg_model, max_faces=10,
            conf_threshold=0.45, landmarks_padding_ratio=1.0,
            inputs=[frame for _, frame in batch_frames.items()],
            include_orig_img=False,
            face_swapper_input_size=(256, 256)  # only here so xseg doesn't resize again
        )

        seg_image_dir = f"{filename_no_etx}\\{self.segment_type}{'_cc' if image_resize_params.center_crop else ''}"

        orig_img, seg_img, control_image_inpaint = self.pre_process_image_for_input(
            frame,
            video_index=0,
            image_resize_params=image_resize_params,
            seg_image_dir=seg_image_dir,
            seg_mask_combined=yolo_results[0][0]['seg_mask_combined'],
            use_only_face_mask=use_only_face_mask,
            use_negative_face_mask=use_negative_face_mask
        )

        # batch_index = 0  # to correct indexing of the yolo results; but since it's an image, we don't need this because they're all the same so we just grab the first one
        for _frame_index, frame in batch_frames.items():

            orig_imgs.append(orig_img.copy())
            seg_imgs.append(seg_img.copy())
            control_images_inpaint.append(control_image_inpaint)
            # batch_index += 1
        print("images", len(orig_imgs), "control_images_inpaint", len(control_images_inpaint))
        return batch_frames, orig_imgs, seg_imgs, control_images_inpaint, yolo_results

    def process_video_for_video(self, video_input_path, filename_no_etx, start_frame, frame_skip, image_resize_params, batch_size, use_only_face_mask, use_negative_face_mask):
        print(f"processing video batch({batch_size}) at frame={start_frame} for {video_input_path}")
        cap = cv2.VideoCapture(video_input_path)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_index = start_frame
        current_index = 0
        batch_frames = {}
        orig_imgs = []
        seg_imgs = []
        control_images_inpaint = []
        while True:
            ret, frame = cap.read()

            if ret:
                if frame_skip == 1 or current_index % frame_skip == 0:
                    # convert to RGB for sd and yolo inputs
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    batch_frames[frame_index] = image_resize_params.apply_params(frame)

            if len(batch_frames) == batch_size or (len(batch_frames) > 0 and not ret):
                # for face segmentation
                # Image.fromarray(batch_frames[0]).show("yolo input")
                yolo_results = extract_predict_yolo8(
                    face_extractor=self.face_extract_model, face_seg_model=self.xseg_model, max_faces=10,
                    conf_threshold=0.45, landmarks_padding_ratio=1.0,
                    inputs=[frame for _, frame in batch_frames.items()],
                    include_orig_img=False,
                    face_swapper_input_size=(256, 256)  # only here so xseg doesn't resize again
                )
                # Image.fromarray(yolo_results[0][0]["aligned_cropped_image"]).show("yolo face output")
                # Image.fromarray(yolo_results[0][0]["seg_mask_combined"]).show("yolo face output")

                batch_index = 0  # to correct indexing of the yolo results
                for _frame_index, frame in batch_frames.items():
                    seg_image_dir = f"{filename_no_etx}\\{self.segment_type}{'_cc' if image_resize_params.center_crop else ''}"

                    orig_img, seg_img, control_image_inpaint = self.pre_process_image_for_input(
                        frame,
                        video_index=_frame_index,
                        image_resize_params=image_resize_params,
                        seg_image_dir=seg_image_dir,
                        seg_mask_combined=yolo_results[batch_index][0]['seg_mask_combined'],
                        use_only_face_mask=use_only_face_mask,
                        use_negative_face_mask=use_negative_face_mask
                    )
                    # debug input
                    # if batch_index == 0:
                    #     orig_img.show("orig img v2v")
                    orig_imgs.append(orig_img.copy())
                    seg_imgs.append(seg_img.copy())
                    control_images_inpaint.append(control_image_inpaint)

                    batch_index += 1
                print("images", len(orig_imgs), "control_images_inpaint", len(control_images_inpaint))
                return batch_frames, orig_imgs, seg_imgs, control_images_inpaint, yolo_results

            frame_index += 1
            current_index += 1

    def process_video_zero(self, filename_no_etx, input_path, control_filename_no_etx, control_input_path, frame_skip, use_opt_flow, batch_size, conditioning_frame_indices, initial_video_control_index, start_frame, temporal_frame_skip, center_crop, seed, max_side, prompt_only, prompt, negative_prompt, num_inference_steps, guidance_scale, clip_skip, strength, controlnet_condition, padding_mask_crop, attach_initial_frame, restore_original_face, use_negative_face_mask, use_only_face_mask, do_single_batch, save_individual_frames):

        output_file_name_format = f"f{frame_skip}-b{batch_size}-r{max_side}{'-xl' if self.use_xl else ''}-{initial_video_control_index}_{start_frame}_{num_inference_steps}_{temporal_frame_skip}_{guidance_scale}_{clip_skip}_{strength}_{controlnet_condition}_{seed}"
        output_path = f"{filename_no_etx}\\f{frame_skip}-b{batch_size}-r{max_side}{'-xl' if self.use_xl else ''}-{initial_video_control_index}_{start_frame}_{num_inference_steps}_{guidance_scale}_{clip_skip}_{strength}_{controlnet_condition}_{seed}.mp4"

        if not os.path.exists(filename_no_etx):
            os.mkdir(filename_no_etx)

        if os.path.exists(output_path):
            while True:
                try:
                    os.remove(output_path)
                    break
                except:
                    print("File in use, waiting 1 second")
                    time.sleep(1)

        # set the control params
        control_cap = cv2.VideoCapture(control_input_path)
        control_fps = control_cap.get(cv2.CAP_PROP_FPS)
        control_w = int(control_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        control_h = int(control_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        control_num_frames = control_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        control_cap.release()
        control_image_resize_params = ImageResizeParams(h=control_h, w=control_w,
                                                        max_side=max_side, center_crop=center_crop)
        print("control_num_frames", control_num_frames, "control_fps", control_fps)

        # set the image/video params
        is_image = input_path.endswith(".png") or input_path.endswith(".jpg") or input_path.endswith(".jpeg") or input_path.endswith(".bmp") or input_path.endswith(".webp") or input_path.endswith(".tiff") or input_path.endswith(".gif")
        if is_image:
            source_image = Image.open(input_path).convert('RGB')
            fps = frame_skip  # leave output fps as 1
            w = source_image.width
            h = source_image.height
            num_frames = 1
        else:
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
        image_resize_params = ImageResizeParams(h=h, w=w,
                                                max_side=max_side, center_crop=center_crop)
        print("num_frames", num_frames, "fps", fps)

        # use control fps if using one image as input
        if is_image:
            output_fps = control_fps / frame_skip
            print("output_fps", output_fps)
        else:
            output_fps = fps / frame_skip
            print("output_fps", output_fps)

        output_video = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mpg2'),
            output_fps,  # fps
            (image_resize_params.new_w, image_resize_params.new_h)
        )

        if seed is None or seed < 0:
            seed = int(time.time())

        print("seed", seed)
        generator = torch.Generator().manual_seed(seed)
        latent_chunks = torch.randn((1, 4, image_resize_params.new_h // 8, image_resize_params.new_w // 8), generator=generator,
                                    dtype=torch.float16)
        # fix latents for all batch frames plus 1 for the control
        latents = latent_chunks.repeat(batch_size+1, 1, 1, 1)

        current_frame_index = start_frame
        current_control_frame_index = initial_video_control_index
        while True:

            control_images_p = self.process_control_video(
                control_filename_no_etx=control_filename_no_etx,
                control_input_path=control_input_path,
                start_frame=current_control_frame_index,
                frame_skip=frame_skip,
                image_resize_params=control_image_resize_params,
                batch_size=batch_size)

            if is_image:
                batch_frames, orig_imgs, seg_imgs, control_images_inpaint, yolo_results = self.process_image_for_video(
                    input_path=input_path,
                    filename_no_etx=filename_no_etx,
                    image_resize_params=image_resize_params,
                    batch_size=batch_size,
                    use_only_face_mask=use_only_face_mask,
                    use_negative_face_mask=use_negative_face_mask
                )
            else:
                batch_frames, orig_imgs, seg_imgs, control_images_inpaint, yolo_results = self.process_video_for_video(
                    video_input_path=input_path,
                    filename_no_etx=filename_no_etx,
                    start_frame=current_frame_index,
                    frame_skip=frame_skip,
                    image_resize_params=image_resize_params,
                    batch_size=batch_size,
                    use_only_face_mask=use_only_face_mask,
                    use_negative_face_mask=use_negative_face_mask
                )

            args = dict(
                prompt=[prompt] * len(orig_imgs),
                negative_prompt=None if negative_prompt is None else [negative_prompt] * len(orig_imgs),
                num_inference_steps=num_inference_steps,
                image=orig_imgs,
                mask_image=seg_imgs,
                guidance_scale=guidance_scale,
                control_image=control_images_p,
                latents=latents,
                height=image_resize_params.new_h,
                width=image_resize_params.new_w
                # padding_mask_crop=padding_mask_crop,
                # eta=1.0,
            )
            if self.use_animate_diff:
                args.pop('latents')
                args.pop('image')
                args.pop('mask_image')
                args.pop('control_image')

                if self.use_animate_diff_v2v:
                    args['video'] = orig_imgs
                else:
                    args["num_frames"] = len(orig_imgs)

                args["prompt"] = prompt
                args["negative_prompt"] = negative_prompt

                if self.use_inpainting:
                    if self.use_multiple_controls:
                        conditioning_frames = [control_images_inpaint, control_images_p]
                    else:
                        conditioning_frames = control_images_inpaint
                else:
                    conditioning_frames = control_images_p
                args["conditioning_frames"] = conditioning_frames
                # args["conditioning_frames"] = [Image.fromarray(np.ones((new_h, new_w, 3), np.uint8)*255) for _ in range(len(orig_imgs))]
                # args["conditioning_frames"] = [control_images_inpaint, control_images_p]

                # args['controlnet_frame_indices'] = [img_idx for img_idx, img in enumerate(conditioning_frame_indices) if img_idx % 4 == 0]
                if self.use_sparsectrl:
                    args["controlnet_frame_indices"] = conditioning_frame_indices
                    args["controlnet_frame_indices"].append(batch_size-1)
                    args["conditioning_frames"] = []
                    print("conditioning_frame_indices", conditioning_frame_indices,
                          len(args["controlnet_frame_indices"]), "orig_imags:", len(orig_imgs))
                    for idx in args["controlnet_frame_indices"]:
                        args["conditioning_frames"].append(orig_imgs[idx])

                    # args["decode_chunk_size"] = 8
                    print("conditioning_frame_indices", conditioning_frame_indices, len(args["controlnet_frame_indices"]), len(args["conditioning_frames"]))

                # if self.context_length > len(orig_imgs):
                #     self.pipeline.disable_free_noise()
                #     self.pipeline.disable_fr
                #
                #     self.pipeline.enable_free_noise(context_length=len(orig_imgs), context_stride=min(4, len(orig_imgs)))
            else:
                if not self.use_inpainting:
                    args["image"] = control_images_p
                if strength is not None:
                    args["strength"] = strength
            if clip_skip > 0:
                args["clip_skip"] = clip_skip
            if controlnet_condition is not None:
                args["controlnet_conditioning_scale"] = controlnet_condition
            if generator is not None:
                args["generator"] = generator

            if prompt_only:
                assert self.use_animate_diff and self.use_inpainting and not self.use_multiple_controls
                args.pop('conditioning_frames')
                args.pop('controlnet_conditioning_scale')
                # args.pop('num_frames')

                if self.use_animate_diff_v2v:
                    args["strength"] = 1.0
                    # args.pop('latents')
                    # args.pop('image')
                    # args.pop('mask_image')
                    # args.pop('control_image')

            # if not self.use_xl:
            #     frame_ids = [initial_video_control_index]
            #     current_index = frame_index-(len(batch_frames)*frame_skip)+frame_skip
            #     for j in range(batch_size):
            #         frame_ids.append(current_index+(j*frame_skip))
            #
            #     args["frame_ids"] = frame_ids
            #     args['video_length'] = len(frame_ids)
            #     args.pop('latents')
            #     args.pop('image')
            #     args.pop('mask_image')
            #     args.pop('control_image')
            #     args.pop('controlnet_conditioning_scale')
            #     args['prompt'] = prompt
            #     print("frame_ids", frame_ids)

            with torch.inference_mode():
                # while True:
                #     try:
                output = self.pipeline(
                    **args
                )
                #         break
                #     except RuntimeError:
                #         print("error, trying again...")
                #         time.sleep(2)
                #         pass


            # post process outputs with yolo_results
            if self.use_animate_diff:
                output_frames = output.frames[0]
            else:
                output_frames = output.images

            final_pil_images = []
            final_cv2_images = []
            chunk_dir = f"{filename_no_etx}\\{frame_skip}-{batch_size}-{max_side}-{seed}{'-cc' if center_crop else ''}"
            if not os.path.exists(chunk_dir):
                os.mkdir(chunk_dir)

            for image_idx, image in enumerate(output_frames):

                # convert to numpy to paste back faces; should be in RGB
                # if image_idx == 0:
                #     image.show("result img v2v")
                np_image = np.array(image)

                # paste each original face back one by one
                if restore_original_face:
                    for (face_index, face_data) in yolo_results[image_idx].items():
                        # no keys = nothing extracted
                        if 'aligned_cropped_image' not in face_data.keys():
                            continue

                        swapped_image = face_data["aligned_cropped_image"]  # unsharp_mask(extracted_data["aligned_cropped_image"], amount=.5)
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
                # for gif "frame_index-(len(batch_frames)*frame_skip)+frame_skip
                if save_individual_frames:
                    image.save(f"{chunk_dir}\\{current_frame_index + (image_idx*frame_skip)}.png")
                final_pil_images.append(image)

                # # convert to numpy for cv2 writer
                # np_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # try:
            if isinstance(final_pil_images[0], np.ndarray):
                _final_pil_images = []
                for img in final_pil_images:
                    _final_pil_images.append(Image.fromarray((img*255).astype(np.uint8)))
                final_pil_images = _final_pil_images
            time_id = time.time()
            # export_to_gif(final_pil_images,
            #               f"{filename_no_etx}\\{current_frame_index}-{frame_skip}-{batch_size}-{max_side}{'-xl' if seg.use_xl else ''}-{initial_video_control_index}_{start_frame}_{num_inference_steps}_{guidance_scale}_{clip_skip}_{strength}_{controlnet_condition}_{seed}.gif",
            #               fps=int(control_fps/frame_skip) if is_image else int(fps/frame_skip))

            _legacy_export_to_video(
                final_pil_images,
                f"{filename_no_etx}\\{current_frame_index}-{frame_skip}-{batch_size}-{max_side}{'-xl' if seg.use_xl else ''}-{initial_video_control_index}_{start_frame}_{num_inference_steps}_{guidance_scale}_{clip_skip}_{strength}_{controlnet_condition}_{seed}.mp4",
                fps=int(control_fps / frame_skip) if is_image else int(fps / frame_skip))

            _legacy_export_to_video(
                final_pil_images,
                f"{chunk_dir}\\{current_frame_index}.mp4",
                fps=int(control_fps/frame_skip) if is_image else int(fps/frame_skip))

            for image_idx, image in enumerate(final_pil_images):
                print("image_idx", image_idx, "image", image, image.info)

            for np_image in final_cv2_images:
                output_video.write(np_image)

            if do_single_batch:
                break

            current_frame_index += batch_size*frame_skip
            current_control_frame_index += batch_size*frame_skip

            if is_image:
                break
            else:
                if current_frame_index >= num_frames:
                    break
                if current_control_frame_index >= control_num_frames:
                    break

        output_video.release()
        run_ffmpeg_optical_flow(output_path, output_path+"-opt.ts", int(fps))

# orig_img, seg_img = segment_image(
#     f_seg_model,
#     "C:\\Users\\teckt\\iae_dfstudio\\yolomodel\\0.jpg")[0]

def segment_image(f_seg_model, img):
    outputs = f_seg_model(img, tracker=False)
    seg_results = []
    for output in outputs:
        # pred = output.plot()
        # pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        # pred = Image.fromarray(pred)
        # pred.show()
        #
        # imgg = output.orig_img
        # imgg = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)



        xy_list = []

        if output.keypoints is not None:
            keypoints = output.keypoints.xy.cpu().detach().numpy()
            boxes_data = output.boxes.data.cpu().detach().numpy()
            boxes_xyxy = boxes_data[:, :4]
            confs = boxes_data[:, -2]

            for idx, xy in enumerate(keypoints):  # tensor of (ids,5,2)
                conf = confs[idx]
                # print("xy", xy, "conf", conf, "boxes_xyxy", boxes_xyxy) # individual face keypoints of shape(5, 2)
                if conf > 0.2:
                    xy_list.append(xy)
        seg_results.append(xy_list)

    return seg_results

def extract_predict_yolo8(face_extractor, face_seg_model, inputs, max_faces, conf_threshold, landmarks_padding_ratio, include_orig_img=False, face_swapper_input_size=(224, 224)):
    _outputs = face_extractor(inputs, tracker=False)  # frames in rgb, convert to PIL if necessary
    post_processing_secs = time.time()

    batch_extracted_new_params = {}
    outputs = []
    for output in _outputs:
        outputs.append(output.cpu().numpy())

    for out_idx, output in enumerate(outputs):
        orig_img = output.orig_img
        orig_img_bw = np.zeros_like(orig_img)[:, :, 0]
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

                aligned_cropped_image, aligned_cropped_params = align_crop_image(
                    original_image=orig_img,
                    original_landmarks=xy,
                    landmarks_padding_ratio=landmarks_padding_ratio
                )

                if aligned_cropped_image.shape[0] >= orig_img.shape[0] or aligned_cropped_image.shape[1] >= orig_img.shape[1]:
                    continue

                try:
                    processed_image = cv2.resize(aligned_cropped_image, face_swapper_input_size, interpolation=cv2.INTER_CUBIC)
                except cv2.error:
                    continue

                if face_seg_model is None:
                    seg_mask = (np.ones_like(aligned_cropped_image) * 255).astype("uint8")
                else:
                    seg_mask = extract_xseg_mask(xseg_model=face_seg_model, face_images=[processed_image])[0]

                batch_extracted_new_params[out_idx][idx] = {
                    "aligned_cropped_image": aligned_cropped_image,
                    "aligned_cropped_params": aligned_cropped_params,
                    "processed_image": processed_image,  # needs to be resized(224x224), rescaled(0,1) image
                    "bbox": boxes_xyxy[idx],
                    "seg_mask": seg_mask,
                    # "orig_img": orig_img.copy() if include_orig_img else None
                }

            # combine face masks
            if len(batch_extracted_new_params[out_idx]) > 0:
                for face_idx, face_data in batch_extracted_new_params[out_idx].items():
                    xyxy = face_data["bbox"]
                    seg_mask = face_data["seg_mask"]
                    h = int(xyxy[3] - xyxy[1])
                    w = int(xyxy[2] - xyxy[0])
                    seg_mask = cv2.resize(seg_mask, (w, h), cv2.INTER_CUBIC)
                    orig_img_bw[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] = seg_mask

                for face_idx, face_data in batch_extracted_new_params[out_idx].items():
                    face_data["orig_img"] = orig_img.copy() if include_orig_img else None
                    face_data["seg_mask_combined"] = orig_img_bw
            else:
                batch_extracted_new_params[out_idx][0] = {
                    "orig_img": orig_img.copy() if include_orig_img else None,
                    "seg_mask_combined": orig_img_bw
                }


    print(f"postprocess face extractor outputs finished in",
          time.time() - post_processing_secs)

    return batch_extracted_new_params


# yyyy = YOLO("yolov8n-face.pt")
# xseg_model = build_seg_model(resolution=(256, 256), load=True, save_path=xseg_model_path)
# tt = time.time()
#
# results = extract_predict_yolo8(
#     face_extractor=yyyy, face_seg_model=xseg_model,max_faces=2, conf_threshold=0.2, landmarks_padding_ratio=1.0,
#     inputs=["ge2.jpg", "ge2.jpg", "ge2.jpg"],
#     include_orig_img=True,
# )
# final_pil_images = []

# for batch_idx, extracted_data in results.items():
#
#     # extracted_data: dict of face data (indexed by face idx) or empty {
#     #                     "aligned_cropped_image": aligned_cropped_image,
#     #                     "aligned_cropped_params": aligned_cropped_params,
#     #                     "processed_image": processed_image,  # needs to be resized(224x224), rescaled(0,1) image
#     #                     "bbox": boxes_xyxy,
#     #                     "seg_mask": seg_mask,
#     #                     "orig_img": orig_img
#     #                 }
#     if not extracted_data:
#         continue
#
#     orig_img = extracted_data[0]["orig_img"]
#     for (face_index, face_data) in extracted_data.items():
#         swapped_image = face_data["aligned_cropped_image"]#unsharp_mask(extracted_data["aligned_cropped_image"], amount=.5)
#         orig_img = paste_swapped_image(
#             dst_image=orig_img,
#             swapped_image=swapped_image,
#             seg_mask=face_data["seg_mask"],
#             aligned_cropped_params=face_data["aligned_cropped_params"],
#             seamless_clone=True,
#             blur_mask=True,
#             resize=False
#         )
#
#     final_pil_images.append(Image.fromarray(orig_img))
#
# for pil_image in final_pil_images:
#     pil_image.show()
#     break
#
# tt = time.time()-tt
# print(1/(tt/3))
# print(results[0][1])

gpu_index = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
import tensorflow as tf


# policy = tf.keras.mixed_precision.Policy("mixed_float16")
# tf.keras.mixed_precision.set_global_policy(policy)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

seg = SegmentInpainter(
    # control_p='normal_bae',
    # control_p='depth_midas',
    control_p='openpose',
    # control_p='canny',
    use_animate_diff=True, use_animate_diff_v2v=False,
    use_lightning=False,
    free_noise_params=[16, 4],
    free_noise_split_params=[256, 16],

    use_xl=False, use_inpainting=True, use_multiple_controls=True, use_sparsectrl=False, segment_type='person' #fashion or person(use negative face mask and open_pose for fun results)
)

seg.set_scheduler(
# scheduler = UniPCMultistepScheduler.from_config(seg.pipeline.scheduler.config)
#     scheduler = DEISMultistepScheduler.from_config(seg.pipeline.scheduler.config)
    scheduler=EulerAncestralDiscreteScheduler.from_config(seg.pipeline.scheduler.config, timestep_spacing="trailing" if seg.use_lightning else 'linspace', beta_schedule="linear")
    # scheduler=DPMSolverMultistepScheduler.from_config(
    #     seg.pipeline.scheduler.config,
    #     # algorithm_type="dpmsolver++",
    #     algorithm_type="sde-dpmsolver++",
    #     solver_order=2,
    #     use_karras_sigmas=True,
    #     timestep_spacing="trailing" if seg.use_lightning else 'linspace'
    # )
)

if USE_ANIMATE_LCM:
    seg.set_scheduler(
        scheduler=LCMScheduler.from_config(seg.pipeline.scheduler.config)
    )

TESTING_BATCH = False

class RedresserOptions:
    def __init__(self, options_dict):
        self.options_dict = options_dict

# ch3: 22 frames
# ch4: 44 frames only one working
# ch5: 112 frames
# bo6: start at 23 or 24 and end at 47 with batch 24 for best results
# bo7: 34 frames
# bo8: 60? frames
video_id = "ge2"
input_path = f"{video_id}.jpg"

control_video_id = "ge2"
control_input_path = f"{control_video_id}.jpg"

cc = np.linspace(0.0, 1.0, 16).tolist()

options_dict = RedresserOptions(
    dict(
        # video name without extension
        filename_no_etx=video_id,
        input_path=input_path,
        control_filename_no_etx=control_video_id,
        control_input_path=control_input_path,

        frame_skip=4,  # free selection
        batch_size=32,  # 2-8
        conditioning_frame_indices=np.arange(0, 16, 4).tolist(),
        max_side=768,  # 640-1024
        seed=-1,
        num_inference_steps=8,  # 15-30, 4-8 for LCM
        attach_initial_frame=False,
        initial_video_control_index=0,  # defaults to 0; controls entire video with this frame
        start_frame=0,  # sets cv2 current position
        temporal_frame_skip=0,  # frames between resetting the video control index; always breaks temporal consistency

        center_crop=True, use_opt_flow=False,
        prompt_only=False,  # must be used together with inpaint and mult=false
        prompt="girl with long hair in red dress"
        negative_prompt='low quality, hands',
        guidance_scale=3,
        clip_skip=0,
        strength=0.90,
        controlnet_condition=.9,  # 1.0 makes it blurry and produces more artifacts
        padding_mask_crop=32,
        restore_original_face=True,
        use_negative_face_mask=True,  # prevents face from being segmented during inpainting
        use_only_face_mask=False,  # only segment face; overrides use_negative_face_mask and ignores all other segments
        do_single_batch=False,  # exit after one batch
        save_individual_frames=False
    )
)

for inference_idx in range(8):
    seg.process_video_zero(
        **options_dict.options_dict,
    )

'''
FOR CROSSATTENTIONFRAMEPROCESSOR, must have:
dpm++ 3m karras, guidance 7+, 20+ steps, control 0.5, make sure unet and control set attn procs
'''

# # inpaint the segmented image
# orig_img, seg_img, control_image = prepare_inpaint_images(orig_img, seg_img)
# # additional control image

# if canny_p is not None:
#     control_image_canny = canny_p(orig_img)
#     control_image = [control_image, control_image_canny]

# generator = torch.Generator("cuda").manual_seed(8008)

# inpainted = []
# for i in range(4):
#     image = pipeline(
#         prompt=prompt, negative_prompt=negative_prompt,
#         image=orig_img, mask_image=seg_img,
#         control_image=control_image,  # comment out remove if not using controlnet inpaint
#         # generator=generator,
#         height=orig_img.height, width=orig_img.width,
#         num_inference_steps=30,
#         guidance_scale=7.5
#         # eta=1.0
#                      ).images[0]
#     image.show()
#     inpainted.append(image)
#
# inpainted.append(orig_img)
# inpainted.append(seg_img)
# grid = make_image_grid(inpainted, rows=2, cols=3)
#
#
# grid.save(f"{time.time()}-inpaint.png")