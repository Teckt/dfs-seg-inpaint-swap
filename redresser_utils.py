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