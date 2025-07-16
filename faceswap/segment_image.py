import os
import numpy as np
import PIL.Image as Image
import cv2




def yolo_segment_image(yolo_model, img, return_original_image=True, preprocess=False):
    """
    draws the mask if exists else draw filled-in boxes
    :param yolo_model:
    :param img: yolo inputs; can be numpy array or a list of them
    :param preprocess (bool): performs BGR2RGB and resizes to seg model input size (320,320)
    :return: a list of tuples containing numpy arrays of the original image(or None) and the mask (3 channels) in 0-255
    """

    if preprocess:
        if isinstance(img, list):
            temp_img = []
            for i, bgr_image in enumerate(img):
                image = cv2.resize(bgr_image, (320, 320), cv2.INTER_CUBIC)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                temp_img.append(image)
            img = temp_img
        else:
            img = cv2.resize(img, (320, 320), cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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


        if output.masks is not None:
            empty_img = np.zeros_like(imgg)
            # mask = mask.astype('uint8')  # Convert to int32 for use with cv2 functions
            # cv2.fillConvexPoly(imgg, [mask], color=(0, 0, 0))  # Black mask
            for mask_idx, mask in enumerate(output.masks.xy):
                conf = output.boxes.conf[mask_idx]
                print("mask", mask.shape, "conf", conf)
                if conf > 0.2:
                    mask = mask.astype(np.int32)  # Convert to int32 for use with cv2 functions
                    cv2.fillPoly(empty_img, [mask], color=(255, 255, 255))  # white mask over black
        elif output.boxes is not None:
            empty_img = np.zeros_like(imgg)
            boxes = output.boxes.xyxy.cpu().numpy()  # Convert tensor to NumPy array if needed

            # Loop over each detected box and draw a white rectangle on the mask
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)  # Convert to integers for drawing
                cv2.rectangle(empty_img, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=-1)
        else:
            empty_img = np.ones_like(imgg)*255

        seg_results.append(
            [
                imgg if return_original_image else None,
                empty_img
            ]
        )

    return seg_results


if __name__ == "__main__":
    from tqdm import tqdm

    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    face_mask_model_path = "yolov8s-face_mask.pt"
    face_mask_model = YOLO(hf_hub_download("Anyfusion/xseg", face_mask_model_path))

    folder = "C:/Users/teckt/documents/test_face_sets/bo"
    out = folder+"/mask"
    os.mkdir(out)

    for file in tqdm([f for f in os.listdir(folder) if f.endswith("jpg") or f.endswith("png")]):
        processed_image = cv2.imread(f"{folder}/{file}")
        _processed_image = cv2.resize(processed_image, (320, 320), cv2.INTER_CUBIC)
        _processed_image = cv2.cvtColor(_processed_image, cv2.COLOR_BGR2RGB)
        _, seg_mask = yolo_segment_image(img=_processed_image, yolo_model=face_mask_model, return_original_image=False)[0]
        Image.fromarray(seg_mask).save(f"{out}/{file}")
