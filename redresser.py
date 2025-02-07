from pipe_manager import SD15PipelineManager
from redresser_utils import *


class RedresserSettings:
    SEGMENT_FASHION = 0
    SEGMENT_PERSON = 1
    SEGMENT_ALL = 2 # replaces whole image unless keep_face or keep_hands is True

    # why are these here? FOR CONVENIENCE
    xseg_model_path="C:/Users/teckt/xseg/models/saved_model"
    face_model_path="yolov8n-face.pt"
    person_model_path="person_yolov8m-seg.pt"
    fashion_model_path="deepfashion2_yolov8s-seg.pt"
    hand_model_path="hand_yolov9c.pt"

    def __init__(self):
        self.SEGMENT_ID = RedresserSettings.SEGMENT_FASHION
        self.keep_hands = False
        self.keep_face = True 
        self.use_faceswap = False
        self.max_side = 1280
        self.center_crop = False
        self.seed = -1

        self.prompt = "A girl in a red dress"
        self.negative_prompt = "bad quality"
        self.num_inference_steps = 20
        self.guidance_scale = 7.5
        self.strength = None
        self.clip_skip = 0
        

class Redresser:

    def __init__(self, local_files_only=True):
        # builds face segmentation model (Tensorflow)
        self.xseg_model = build_seg_model(resolution=(256, 256), load=True, save_path=RedresserSettings.xseg_model_path)
        # builds face detection model from YOLO (PyTorch)
        self.face_extract_model = YOLO(RedresserSettings.face_model_path)
        # initialize default settings first
        self.settings = RedresserSettings()
        # initialize the sd pipeline
        self.pipe_manager = SD15PipelineManager(local_files_only=local_files_only)

    def set_seg_models(self,):
        '''
        loads the models for face seg, face detect, fashion/person seg, hand seg
        '''
        if self.settings.SEGMENT_ID not in [RedresserSettings.SEGMENT_FASHION, RedresserSettings.SEGMENT_PERSON, RedresserSettings.SEGMENT_ALL]:
            print("segment_type does not exist:", self.settings.SEGMENT_ID)
            assert False

        # segments images; black fill
        if self.settings.SEGMENT_ID == RedresserSettings.SEGMENT_PERSON:
            self.f_seg_model = YOLO(hf_hub_download("Bingsu/adetailer", RedresserSettings.person_model_path))
        elif self.settings.SEGMENT_ID == RedresserSettings.SEGMENT_FASHION:
            self.f_seg_model = YOLO(hf_hub_download("Bingsu/adetailer", RedresserSettings.fashion_model_path))
        else:
            print("Segment not implemented", self.settings.SEGMENT_ID)
            assert False

        if self.keep_hands:
            self.hand_seg_model = YOLO(hf_hub_download("Bingsu/adetailer", RedresserSettings.hand_model_path))
        
    def run(self, image_path, settings={}):
        '''
        settings(dict): run with redresser and pipe settings
            -default pipe settings (will only update these keys if you have them)
            "mode": SD15PipelineManager.USE_IMAGE (0),
            "use_LCM": True, # only used for video pipe
            "scheduler": "EulerAncestralDiscreteScheduler",
            "use_inpaint_control_net": True,
            "control_net_id": 'openpose'
            -default redresser settings
        '''
        batch_size = 1 # since we're not doing video, just process 1 image

        # update settings
        self.pipe_manager.apply_settings(settings)
        self.pipe_manager.set_pipeline()
        self.set_seg_models()

        batch_frames, orig_imgs, seg_imgs, control_images_inpaint, control_images_p, yolo_results, image_resize_params \
            = self.prepare_inputs(image_path, self.settings.max_side, self.settings.center_crop)
        
        seed = self.settings.seed
        if seed is None or seed < 0:
            seed = int(time.time())

        print("seed", seed)
        generator = torch.Generator().manual_seed(seed)

        args = dict(
            prompt=self.settings.prompt,
            
            num_inference_steps=self.settings.num_inference_steps,
            
            image=orig_imgs,
            mask_image=seg_imgs,

            guidance_scale=self.settings.guidance_scale,
            height=image_resize_params.new_h,
            width=image_resize_params.new_w
        )
        if self.pipe_manager.pipe_settings.get("use_inpaint_control_net", True):
            args["control_image"] = [control_images_inpaint, control_images_p]
        else:
            args["image"] = control_images_p

        if self.settings.negative_prompt is not None:
            args["negative_prompt"] = self.settings.negative_prompt
        if self.settings.strength is not None:
            args["strength"] = self.settings.strength
        if self.settings.clip_skip > 0:
            args["clip_skip"] = self.settings.clip_skip
        if generator is not None:
            args["generator"] = generator

        with torch.inference_mode():
            output = self.pipeline(
                **args
            )

        final_pil_images = self.process_outputs(output.images, yolo_results)

        time_id = time.time()

        for image_idx, image in enumerate(final_pil_images):
            print("image_idx", image_idx, "image", image, image.info)
            image.save(f"{time_id}_{str(image_idx).zfill(5)}.png")

    def process_outputs(self, outputs, yolo_results):
        final_pil_images = []
        final_cv2_images = []

        for image_idx, image in enumerate(outputs):

            np_image = np.array(image)

            # paste each original face back one by one
            if self.keep_face:
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
            final_pil_images.append(image)
        
        return final_pil_images

    def prepare_inputs(self, image_path, max_side, center_crop):
        source_image = Image.open(image_path).convert('RGB')
        w = source_image.width
        h = source_image.height
        image_resize_params = ImageResizeParams(
            h=h, w=w, max_side=max_side, center_crop=center_crop)
        frame = image_resize_params.apply_params(frame)

        print(f"processing image batch({0}) at {image_path}")

        batch_frames = {}
        orig_imgs = []
        seg_imgs = []
        control_images_inpaint = []
        for i in range(1):
            batch_frames[i] = frame

        # face segmentation
        yolo_results = yolo8_extract_faces(
            face_extractor=self.face_extract_model, face_seg_model=self.xseg_model, max_faces=10,
            conf_threshold=0.45, landmarks_padding_ratio=1.0,
            inputs=[frame for _, frame in batch_frames.items()],
            include_orig_img=False,
            face_swapper_input_size=(256, 256)  # only here so xseg doesn't resize again
        )

        # redresser segmentation
        orig_img, seg_img, control_image_inpaint = self.segment_image(
            frame,
            video_index=0,
            image_resize_params=image_resize_params,
            seg_image_dir="seg",
            seg_mask_combined=yolo_results[0][0]['seg_mask_combined'],
        )

        # batch_index = 0  # to correct indexing of the yolo results; but since it's an image, we don't need this because they're all the same so we just grab the first one
        for _frame_index, frame in batch_frames.items():

            orig_imgs.append(orig_img.copy())
            seg_imgs.append(seg_img.copy())
            control_images_inpaint.append(control_image_inpaint)
            # batch_index += 1
        print("images", len(orig_imgs), "control_images_inpaint", len(control_images_inpaint))

        control_images_p = [self.pipe_manager.control_p(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))]
        
        return batch_frames, orig_imgs, seg_imgs, control_images_inpaint, control_images_p, yolo_results, image_resize_params

        
    def segment_image(self, image, video_index, image_resize_params, seg_image_dir, seg_mask_combined):
        '''
        seg_mask_combined: the combined segmented face masks for this image
        '''
        
        # image should be in RGB
        if seg_image_dir is not None:
            if not os.path.exists(seg_image_dir):
                os.mkdir(seg_image_dir)

            seg_path = f"{seg_image_dir}\\{str(video_index).zfill(5)}.png"

        if seg_image_dir is None or not os.path.exists(seg_path):
            # process and segment the image
            orig_img, seg_img = self.segment_image(image)[0]
            orig_img = orig_img.resize((image_resize_params.new_w, image_resize_params.new_h))
            if seg_image_dir is not None:
                seg_img.save(seg_path)
        else:
            # load the segmentation from local disk instead
            orig_img = Image.fromarray(image).resize((image_resize_params.new_w, image_resize_params.new_h)).convert('RGB')
            seg_img = Image.open(seg_path).convert('RGB')

        seg_img = seg_img.resize((image_resize_params.new_w, image_resize_params.new_h))
        if seg_mask_combined is not None:
            if self.segment_type == Redresser.SEGMENT_ALL:  # just all white
                seg_img = np.array(seg_img, dtype=np.uint8)
                seg_img[face_mask > -1] = 1

            if self.keep_face:  # removes faces from seg; used to keep faces intact or for face restore
                face_mask = seg_mask_combined
                seg_img = np.array(seg_img, dtype=np.uint8)
                seg_img[face_mask > 127] = 0

            seg_img = Image.fromarray(seg_img).convert('RGB')

        # prepare to inpaint the segmented image
        control_image_inpaint = make_inpaint_condition(orig_img, seg_img, 0.5)

        return orig_img, seg_img, control_image_inpaint
    

