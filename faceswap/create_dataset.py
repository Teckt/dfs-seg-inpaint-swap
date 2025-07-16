import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from ultralytics import YOLO
import os

from dfs_seg_inpaint_swap.tf_free_functions import align_crop_image


def from_image_dir(image_dir, output_dir, max_faces=4, landmarks_padding_ratio=1.0, create_eye_mask=False):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    face_extractor = YOLO(hf_hub_download("Anyfusion/xseg", "yolov8n-face.pt"))
    files = [f"{image_dir}/{f}" for f in os.listdir(image_dir) if f.endswith(".jpeg") or f.endswith(".jpg") or f.endswith(".png") or f.endswith(".webp")]

    yolo_gen = face_extractor(files, stream=True)  #, vid_stride=30)
    frame_index = 0
    for result in tqdm(yolo_gen):
        keypoints = result.keypoints.xy.cpu().numpy()
        keypointsn = result.keypoints.xyn.cpu().numpy()
        boxes_data = result.boxes.data.cpu().numpy()
        frame = result.orig_img
        if len(keypoints) > 0:

            confs = boxes_data[:, -2]
            for idx, xy in enumerate(keypoints):  # tensor of (ids,5,2)
                if idx >= len(confs):
                    continue
                if confs[idx] < 0.3:
                    continue

                if idx + 1 > max_faces:
                    break

                try:
                    aligned_cropped_image, aligned_cropped_param = align_crop_image(
                        original_image=frame,
                        original_landmarks=xy,
                        landmarks_padding_ratio=landmarks_padding_ratio
                    )
                except:
                    continue

                # verify if out of bounds
                if aligned_cropped_image.shape[0] >= max(frame.shape[:2]) or \
                        aligned_cropped_image.shape[1] >= max(frame.shape[:2]):
                    continue

                if create_eye_mask:
                    leye, reye = aligned_cropped_param["cropped_landmarks"][:2, :]
                    eye_mask = np.zeros_like(aligned_cropped_image)
                try:
                    # basename = os.path.basename(video_path)
                    # cv2.imwrite(
                    #     f"{aligned_faces_dir}/{basename}_{str(frame_index).zfill(5)}_{str(idx).zfill(1)}.jpg",
                    #     aligned_cropped_image)

                    basename = os.path.basename(files[frame_index])
                    cv2.imwrite(f"{output_dir}/{basename}_{str(idx).zfill(1)}.jpg", aligned_cropped_image)
                except:
                    pass

        frame_index += 1

def from_video_dir(input_dir, output_dir, max_faces=4, landmarks_padding_ratio=1.0):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    face_extractor = YOLO(hf_hub_download("Anyfusion/xseg", "yolov8n-face.pt"))
    videos = [f"{input_dir}/{f}" for f in os.listdir(input_dir) if f.endswith(".mp4") or f.endswith(".ts") or f.endswith(".webm") or f.endswith(".mkv")]
    for video_idx, video in enumerate(videos):
        yolo_gen = face_extractor(video, stream=True, vid_stride=30)
        frame_index = 0
        for result in tqdm(yolo_gen):
            keypoints = result.keypoints.xy.cpu().numpy()
            keypointsn = result.keypoints.xyn.cpu().numpy()
            boxes_data = result.boxes.data.cpu().numpy()
            frame = result.orig_img
            if len(keypoints) > 0:

                confs = boxes_data[:, -2]
                for idx, xy in enumerate(keypoints):  # tensor of (ids,5,2)
                    if idx >= len(confs):
                        continue
                    if confs[idx] < 0.3:
                        continue

                    if idx + 1 > max_faces:
                        break

                    try:
                        aligned_cropped_image, aligned_cropped_param = align_crop_image(
                            original_image=frame,
                            original_landmarks=xy,
                            landmarks_padding_ratio=landmarks_padding_ratio
                        )
                    except:
                        continue

                    # verify if out of bounds
                    if aligned_cropped_image.shape[0] >= max(frame.shape[:2]) or \
                            aligned_cropped_image.shape[1] >= max(frame.shape[:2]):
                        continue
                    try:
                        basename = os.path.basename(videos[video_idx])
                        cv2.imwrite(f"{output_dir}/{basename}_{str(frame_index).zfill(5)}_{str(idx).zfill(1)}.jpg", aligned_cropped_image)
                    except:
                        pass

            frame_index += 1


if __name__ == "__main__":
    # from_image_dir(
    #     image_dir="C:/Users/teckt/Documents/s783",
    #     output_dir="C:/Users/teckt/Documents/test_face_sets/s7",
    #     max_faces=4,
    #     landmarks_padding_ratio=1.0
    # )
    from_video_dir(
        input_dir="C:/Users/teckt/Documents/webcam",
        output_dir="C:/Users/teckt/Documents/test_vids/F/aligned",
        max_faces=4,
        landmarks_padding_ratio=1.0
    )