import multiprocessing
import os
import subprocess

import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from torchvision.transforms import transforms
from ultralytics import YOLO

import queue as queue_errors
import random
import time

import cv2

from dfs_seg_inpaint_swap.faceswap.segment_image import yolo_segment_image
from dfs_seg_inpaint_swap.faceswap_model import FaceSwapModel, FaceSwapModel2x
from dfs_seg_inpaint_swap.tf_free_functions import fit_to_size, align_crop_image, paste_swapped_image, color_transfer, get_face_center

import numpy as np

xseg_model_path = "C:/Users/teckt/PycharmProjects/iae_dfstudio/models/xseg/saved_model"
face_swapper_input_size = (224, 224)
face_extractor_input_size = (416, 416)
frame_annotation_folder = "C:/Users/teckt/Documents/webcam/_frames"

def run_ffmpeg_optical_flow(input_video: str, output_video: str, fps: int = 60):
    """
    Run FFmpeg's minterpolate filter to calculate optical flow and interpolate frames.

    Args:
        input_video (str): Path to the input video file.
        output_video (str): Path to the output video file.
        fps (int): The target frames per second (default is 60).
    """
    # Define the FFmpeg command with minterpolate filter
    if "gif" in output_video[:-4]:
        ffmpeg_command = [
            'ffmpeg',
            '-y',  # Overwrite the output file if it exists
            '-i', input_video,  # Input video
            '-vf', f"minterpolate='fps={fps}'",  # Video filter for optical flow
            # '-preset', 'veryslow',  # Use slower, better compression for quality
            # '-crf', '17',  # Constant Rate Factor, lower is better quality (18-23 is good)
            output_video  # Output video file
        ]
    else:
        ffmpeg_command = [
            'ffmpeg',
            '-y',  # Overwrite the output file if it exists
            '-i', input_video,  # Input video
            '-vf', f"minterpolate='fps={fps}'",  # Video filter for optical flow
            '-c:v', 'libx264',  # Use H.264 codec for video
            # '-b:v', '5000k',  # Set bitrate to 5000 kbps (adjust as needed)
            '-preset', 'veryslow',  # Use slower, better compression for quality
            '-crf', '17',  # Constant Rate Factor, lower is better quality (18-23 is good)
            output_video  # Output video file
        ]

    try:
        # Run the FFmpeg command
        result = subprocess.run(ffmpeg_command, check=True, text=True, capture_output=True)
        print("FFmpeg Output:", result.stdout)
        print("FFmpeg Error (if any):", result.stderr)
        print(f"Successfully processed video: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg command failed with error: {e.stderr}")


def watermark_frame(frame):
    # Define the text and font properties
    text = "AI generated in Deepfake Studio"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 * max(frame.shape[:2])/1000
    font_color = (255, 255, 255)  # White color
    font_thickness = 1

    # Get the width and height of the text box
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Calculate the bottom right corner position for the text
    x = frame.shape[1] - text_width - 20  # 10 pixels from the right edge
    y = frame.shape[0] - baseline - 20  # 10 pixels from the bottom edge

    # Write the text on the image
    cv2.putText(frame, text, (x, y), font, font_scale, font_color, font_thickness)

    return frame


def read_frames_into_queue(video_file_path, stop_signal_queue, output_queue, batch_size=32, max_frame_size=720, secs_to_stop_at=None, exit_on_complete=False):
    print(f"read_frames_into_queue started, batch_size={batch_size}, max_frame_size={max_frame_size}, {video_file_path}")
    wait = 0
    # while True:
    #     if os.path.exists(video_file_path):
    #         try:
    #             os.rename(video_file_path, video_file_path)
    #             break
    #         except OSError as e:
    #             print(f"Error reading from path. Waiting({wait})")
    #             wait += 1
    #             time.sleep(1)
    #     else:
    #         print(f"Path does not exist. Waiting({wait})")
    #         wait += 1
    #         time.sleep(1)

    cap = cv2.VideoCapture(video_file_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    frames_to_stop_at = fps*600 if secs_to_stop_at is None else fps*secs_to_stop_at  # 10 minutes of frames

    # frame_skip_interval = 1/(fps/24)
    # frame_skip_counter = 0
    # use_frame_skip = fps > 24

    frame_index = 0
    if cap is None:
        output_queue.put(None)
        return

    queue_index = 0

    original_frames = []

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_side = max(w, h)
    if max_side > max_frame_size:
        ratio = max_frame_size / max_side
        w = int(min(ratio * w, max_frame_size))
        h = int(min(ratio * h, max_frame_size))
    else:
        ratio = 1

    print("ratio", ratio, "w", w, "h", h)

    while True:

        try:
            stop_signal = stop_signal_queue.get(block=False)
            if stop_signal is None:
                output_queue.put(None, block=True)
                print("stopping read_frames_into_queue")
                break
        except queue_errors.Empty:
            pass

        start_time = time.time()
        ret, frame = cap.read()

        if frame_index < frames_to_stop_at and ret:



            # timestamp = frame_to_timestamp(frame_index, cap.get(cv2.CAP_PROP_FPS))
            frame_index += 1

            # if use_frame_skip and frame_skip_counter >= 1:
            #     frame_skip_counter -= 1
            #     continue

            original_frames.append(frame)
            # pre_processed_inputs.append(pre_processed_input)

            # if use_frame_skip:
            #     frame_skip_counter += frame_skip_interval

            if len(original_frames) < batch_size:
                continue

            resized_frames, pre_processed_inputs = video_reader_pre_process_for_face_extractor(ratio, original_frames, w, h)
            print(
                f"frames preprocessed [queue_index={queue_index}, batch_size={batch_size}, frame_index={frame_index}, frame.shape={resized_frames[0].shape}, ] took {(time.time() - start_time):.2f} secs")
            # refreshes the frame queue
            output_queue.put(
                (queue_index, np.array(resized_frames), np.array(pre_processed_inputs), video_file_path, fps),
                block=True)

            original_frames.clear()
            queue_index += 1

        else:
            resized_frames, pre_processed_inputs = video_reader_pre_process_for_face_extractor(ratio, original_frames, w, h)
            if len(resized_frames) > 0 and len(pre_processed_inputs) > 0:
                print(
                    f"frames preprocessed [queue_index={queue_index}, batch_size={batch_size}, frame_index={frame_index}, frame.shape={resized_frames[0].shape}, ] took {(time.time() - start_time):.2f} secs")
                # refreshes the frame queue
                output_queue.put(
                    (queue_index, np.array(resized_frames), np.array(pre_processed_inputs), video_file_path, fps),
                    block=True)
            print(f"stream reached end of frames, put None, [queue_index={queue_index}, batch_size={batch_size}, frame_index={frame_index}, len(resized_frames)={len(resized_frames)}, ]")
            output_queue.put(None, block=True)

            if exit_on_complete:
                break

            time.sleep(1)
            continue


def video_reader_pre_process_for_face_extractor(ratio, original_frames, w, h):
    # place into queues to process
    # print(f"placing {len(original_frames)} frames to pre process queues")
    resized_frames = []
    pre_processed_inputs = []
    for frame in original_frames:
        if ratio != 1:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frames.append(np.asarray(frame, dtype='uint8'))
        preprocessed_frame = fit_to_size(frame, (416, 416), dtype='uint8', norm=False)
        pre_processed_inputs.append(preprocessed_frame)
    return resized_frames, pre_processed_inputs


def get_frame_interval(fps):
    rand_int = random.randint(-fps, fps)
    return int((fps*5)+rand_int*.75)


def extract_faces(input_queue, output_queue, merge_output_queue, max_faces, conf_threshold, landmarks_padding_ratio, detect_process_only_input_queue, detect_process_only_output_queue):
    req_save_frames_for_annotate=True

    face_extractor = YOLO("C:\\Users\\teckt\\PycharmProjects\\iae_dfstudio\\yolomodel\\yolov8n-face.pt")
    # face_extractor = YOLO(hf_hub_download("Anyfusion/xseg", "yolov8n-face.pt"))

    frame_annotate_last_index = 0
    total_frames_processed = 0
    while True:

        try:
            queue_inputs = input_queue.get(block=False)
        except queue_errors.Empty:
            continue

        if queue_inputs is None:
            # empty queue
            print("extractor none received")
            while True:
                try:
                    input_queue.get(block=False)
                except queue_errors.Empty:
                    print("extractor put None in output")
                    output_queue.put(None, block=True)
                    break

            # reset the frames indexes
            frame_annotate_last_index = 0
            total_frames_processed = 0

            continue

        queue_index, original_frames, batch_pre_processed_inputs, video_file_path, fps = queue_inputs
        secs = time.time()
        fps = int(fps)

        frame_annotate_save_interval = get_frame_interval(fps)

        extract_predict_outputs = extract_predict_yolo8(
                face_extractor,
                queue_index,
                original_frames,
                batch_pre_processed_inputs,
                total_frames_processed,
                max_faces,
                conf_threshold,
                landmarks_padding_ratio,
                req_save_frames_for_annotate,
                frame_annotate_last_index,
                frame_annotate_save_interval,
                fps,
                video_file_path,
                detect_process_only_input_queue,
                detect_process_only_output_queue
        )

        batch_processed_face_swap_input_images, \
            batch_extracted_images, \
            batch_extracted_params,\
            batch_extracted_indexes, \
            batch_extrapolated_bboxes, \
            batch_extracted_new_params, \
            total_frames_processed, \
            req_save_frames_for_annotate, \
            frame_annotate_last_index, \
            frame_annotate_save_interval, \
            fps \
            = extract_predict_outputs

        if len(batch_processed_face_swap_input_images) == 0:
            print(f"no faces detected [queue_index={queue_index}] finished in", time.time() - secs)

            if req_save_frames_for_annotate and (frame_annotate_last_index == 0 or total_frames_processed-frame_annotate_last_index > frame_annotate_save_interval):
                # save image to annotation folder
                frame_annotate_last_index = total_frames_processed + 1 - 1
                frame_annotate_save_interval = get_frame_interval(fps)

                # save_image_for_annotate(original_frames[random.randint(0, len(original_frames)-1)], video_file_path, total_frames_processed)

            for i, frame in enumerate(original_frames):
                merge_output_queue.put((queue_index, i, frame, None, 0, 0))

            # del filtered_pred_bboxes
            continue

        print(f"total face extraction [queue_index={queue_index}] took in", time.time() - secs)

        output_queue.put((
            queue_index,
            original_frames,
            batch_processed_face_swap_input_images,
            batch_extracted_images,
            batch_extracted_params,
            batch_extracted_indexes,
            batch_extrapolated_bboxes,
            batch_extracted_new_params
            # batch_post_processed_bboxes.copy()
        ), block=True)
        # del filtered_pred_bboxes

def detect_process_only(detect_process_only_input_queue, detect_process_only_output_queue):
    face_extractor = YOLO(hf_hub_download("Anyfusion/xseg", "yolov8n-face.pt"))
    face_extractor.to("cuda:0")

    while True:

        original_frames = detect_process_only_input_queue.get(block=True)
        print("received input tensors")
        with torch.no_grad():
            pil_frames = [Image.fromarray(frame).convert('RGB') for frame in original_frames]
            # pil_frames[0].show(f"{total_frames_processed}-{queue_index}")

            _outputs = face_extractor.predict(pil_frames, tracker=False, conf=0.3,
                                              half=False, stream=True)  # frames in rgb, convert to PIL if necessary
            # outputs = []
            # for result in _outputs:
            #     outputs.append(result.cpu().numpy())

            outputs = []
            for result in _outputs:
                keypoints = result.keypoints.xy.cpu().numpy()
                keypointsn = result.keypoints.xyn.cpu().numpy()
                boxes_data = result.boxes.data.cpu().numpy()
                outputs.append({"keypoints": keypoints, "boxes_data": boxes_data, "keypointsn": keypointsn})

        detect_process_only_output_queue.put(outputs, block=True)

def extract_predict_yolo8(face_extractor, queue_index, original_frames, batch_pre_processed_inputs, total_frames_processed, max_faces, conf_threshold, landmarks_padding_ratio, req_save_frames_for_annotate, frame_annotate_last_index, frame_annotate_save_interval, fps, video_file_path, detect_process_only_input_queue, detect_process_only_output_queue):
    secs = time.time()
    # detect_process_only_input_queue.put(original_frames, block=True)
    # outputs = detect_process_only_output_queue.get(block=True)


    # pil_frames = [Image.fromarray(frame).convert('RGB') for frame in original_frames]
    pil_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in original_frames]
    _outputs = face_extractor.predict(pil_frames, tracker=False, conf=conf_threshold,
                                      half=False, stream=True)  # frames in rgb, convert to PIL if necessary
    # outputs = _outputs
    outputs = []
    for result in _outputs:
        keypoints = result.keypoints.xy.cpu().numpy()
        keypointsn = result.keypoints.xyn.cpu().numpy()
        boxes_data = result.boxes.data.cpu().numpy()
        outputs.append({"keypoints": keypoints, "boxes_data": boxes_data, "keypointsn": keypointsn})
    print(f"detect {time.time()-secs}")

    post_processing_secs = time.time()
    # put preds in batches
    batch_processed_face_swap_input_images = []
    # batch_post_processed_bboxes = []
    batch_extracted_images = []
    batch_extracted_params = []
    batch_extracted_indexes = []
    batch_extrapolated_bboxes = []  # only one that is grouped by batch

    batch_extracted_new_params = {}

    for out_idx, result in enumerate(outputs):

        # imgg = output.orig_img
        # imgg = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)
        # xy_list = []

        if len(result) > 0:
            batch_extracted_new_params[out_idx] = {}
            # if use_tensor:

            keypointsn = result['keypointsn']
            keypoints = result['keypoints']
            boxes_data = result['boxes_data']
            # else:
            #     keypoints = result.keypoints.xy
            #     # keypointsn = result.keypoints.xyn
            #     boxes_data = result.boxes.data

            boxes_xyxy = boxes_data[:, :4]
            confs = boxes_data[:, -2]
            batch_extrapolated_bboxes.append(boxes_xyxy)
            # test_frame = original_frames[out_idx].copy()
            for idx, xy in enumerate(keypoints):  # tensor of (ids,5,2)
                # for (x,y) in xy:
                #     cv2.circle(test_frame, (int(x), int(y)), 7, (0, 0, 255))
                if idx >= len(confs):
                    continue
                if confs[idx] < 0.3:
                    continue

                if idx + 1 > max_faces:
                    break

                the_time = time.time()

                # # see if size can be further reduced before processing to speed things up
                # face_center, dx, dy, qsize = get_face_center(xy)
                # # get the ratio from qsize to img resolution and scale down if bigger than face swapper input size
                # if qsize < face_swapper_input_size[0]:
                #     ratio = face_swapper_input_size[0] / qsize
                #     h, w = original_frames[out_idx].shape[:2]
                #     new_h, new_w = int(ratio*h), int(ratio*w)
                #     xyn = keypointsn[idx]
                #     new_xy = xyn.copy()
                #     new_xy[..., 0] = new_xy[..., 0] * new_w
                #     new_xy[..., 1] = new_xy[..., 1] * new_h
                #     aligned_cropped_image, aligned_cropped_params = align_crop_image(
                #         original_image=cv2.resize(original_frames[out_idx], (new_w, new_h), interpolation=cv2.INTER_CUBIC),
                #         original_landmarks=new_xy,
                #         landmarks_padding_ratio=landmarks_padding_ratio
                #     )
                #     # convert the params back to original landmarks by using the norm
                #     aligned_cropped_params["original_landmarks"] = aligned_cropped_params["original_landmarks"]/ ratio
                #     aligned_cropped_params["rotated_landmarks"] = aligned_cropped_params["rotated_landmarks"]/ ratio
                #     aligned_cropped_params["crop_box"] = aligned_cropped_params["crop_box"]/ ratio
                #     aligned_cropped_params["cropped_landmarks"] = aligned_cropped_params["cropped_landmarks"]/ ratio
                # else:
                try:
                    aligned_cropped_image, aligned_cropped_params = align_crop_image(
                        original_image=original_frames[out_idx],
                        original_landmarks=xy,
                        landmarks_padding_ratio=landmarks_padding_ratio
                    )
                except:
                    continue

                # print(f"[queue_index={queue_index}] align:", time.time() - the_time)

                the_time = time.time()
                if aligned_cropped_image.shape[0] >= max(original_frames[out_idx].shape[:2]) or aligned_cropped_image.shape[1] >= max(original_frames[out_idx].shape[:2]):
                    if req_save_frames_for_annotate and idx != 0 and (
                            total_frames_processed - frame_annotate_last_index > frame_annotate_save_interval):
                        frame_annotate_last_index = total_frames_processed + 1 - 1
                        frame_annotate_save_interval = get_frame_interval(fps)
                        # save_image_for_annotate(original_frames[0], video_file_path, total_frames_processed)
                    continue

                try:
                    processed_image = cv2.resize(aligned_cropped_image, face_swapper_input_size,
                                                 interpolation=cv2.INTER_CUBIC)
                except cv2.error:
                    continue

                batch_processed_face_swap_input_images.append(processed_image)
                batch_extracted_images.append(aligned_cropped_image)
                batch_extracted_params.append(aligned_cropped_params)
                batch_extracted_indexes.append(out_idx)
                # batch_extrapolated_bboxes.append(box)
                # batch_post_processed_bboxes.append(bboxes)
                batch_extracted_new_params[out_idx][idx] = {
                    "aligned_cropped_image": aligned_cropped_image,
                    "aligned_cropped_params": aligned_cropped_params,
                    "processed_image": processed_image,  # needs to be resized(224x224), rescaled(0,1) image
                }
                # print(f"[queue_index={queue_index}] the rest:", time.time() - the_time)
            # Image.fromarray(test_frame).show()
            # key should not exist in output
            if len(batch_extracted_new_params[out_idx]) == 0:

                batch_extrapolated_bboxes[out_idx] = None
                del batch_extracted_new_params[out_idx]



        else:
            batch_extrapolated_bboxes.append(None)
            if req_save_frames_for_annotate and (
                    frame_annotate_last_index == 0 or total_frames_processed - frame_annotate_last_index > frame_annotate_save_interval):
                # save image to annotation folder
                frame_annotate_last_index = total_frames_processed + 1 - 1
                frame_annotate_save_interval = get_frame_interval(fps)

                # save_image_for_annotate(original_frames[random.randint(0, len(original_frames) - 1)],
                #                         video_file_path, total_frames_processed)

        total_frames_processed += 1

    print(f"postprocess face extractor outputs [queue_index={queue_index}] finished in",
          time.time() - post_processing_secs)

    return batch_processed_face_swap_input_images, \
        batch_extracted_images, \
        batch_extracted_params,\
        batch_extracted_indexes, \
        batch_extrapolated_bboxes, \
        batch_extracted_new_params, \
        total_frames_processed, \
        req_save_frames_for_annotate, \
        frame_annotate_last_index, \
        frame_annotate_save_interval, \
        fps


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def swap_process_only(checkpoint_queue, swap_process_only_input_queue, swap_process_only_output_queue):
    from dfs_seg_inpaint_swap.faceswap_model import FaceSwapModel
    # load face swap model
    image_size = 320
    latent_dim = 512
    fs_model = FaceSwapModel2x(
        (3, image_size, image_size),
        dims=latent_dim,
        encoder="", decoder=""
    )
    fs_model.to("cuda:0")

    while True:
        try:
            checkpoint_path = checkpoint_queue.get(block=False)

            change_checkpoint(fs_model, checkpoint_path)
            fs_model.to("cuda:0")
            print("checkpoint loaded")
        except queue_errors.Empty:
            pass

        input_tensors = swap_process_only_input_queue.get(block=True)
        print("received input tensors")
        with torch.no_grad():
            result_tensors = fs_model.predict(input_tensors.to("cuda:0"))

        swap_process_only_output_queue.put(result_tensors.to("cpu"), block=True)

def swap_faces_multi_face_fix(input_queue, merge_input_queues, merge_output_queue, set_segmentation_mask_queue, swap_process_only_input_queue, swap_process_only_output_queue):
    from dfs_seg_inpaint_swap.faceswap_model import FaceSwapModel
    from ultralytics import YOLO

    face_mask_model_path = "yolov11s-seg.pt"
    face_mask_model = YOLO(hf_hub_download("Anyfusion/xseg", face_mask_model_path))

    # load face swap model
    image_size = 320
    latent_dim = 512
    fs_model = FaceSwapModel(
        (3, image_size, image_size), dims=latent_dim,
        encoder="", decoder=""
    )

    use_segmentation_mask = True

    first_run = True

    while True:
        # swap video
        try:
            queue_inputs = input_queue.get(block=False)
        except queue_errors.Empty:
            continue

        if queue_inputs is None:
            # empty queue
            while True:
                try:
                    input_queue.get(block=False)
                except queue_errors.Empty:
                    break

            while True:
                try:
                    set_segmentation_mask_queue.get(block=False)
                except queue_errors.Empty:
                    break

            break_process = False

            print("swap process none received")
            first_run = True

            # this is in testing for videoswapper swap, use continue for webcamstreamer
            if break_process:
                print("breaking swap process for next job")
                break
            else:
                for q in merge_input_queues:
                    q.put(None, block=True)
                print("ending/continuing swap process for next job")
                continue

        queue_index, \
        original_frames, \
        batch_processed_face_swap_input_images, \
        batch_extracted_images, \
        batch_extracted_params, \
        batch_extracted_indexes, \
        batch_extrapolated_bboxes, \
        batch_extracted_new_params = queue_inputs

        try:
            # no idea if this would set set_segmentation_mask_queue_output to null if it fails so we'll just avoid it by assigning this twice
            set_segmentation_mask_queue_output = set_segmentation_mask_queue.get(block=False)

            use_segmentation_mask = set_segmentation_mask_queue_output
        except queue_errors.Empty:
            pass

            # send None signal to stop next process

        total_secs = time.time()

        swap_secs = time.time()

        batch_size = len(original_frames)

        current_batch_index = 0

        merge_batch = {}

        if first_run:
            print(f"swap_faces got [queue_index={queue_index}, current_batch_index={current_batch_index}] got", len(batch_processed_face_swap_input_images))

        # put them all in a real batch
        face_batch_input = []
        mask_batch_input = []
        for (batch_index, faces_data) in batch_extracted_new_params.items():
            if not faces_data:
                continue
            for (face_index, face_data) in faces_data.items():
                # processed_image = np.clip(face_data["processed_image"] / 255., 0., 1.)
                processed_image = fs_model.transform_inference_inputs(face_data["processed_image"])

                face_batch_input.append(processed_image)
                mask_batch_input.append(face_data["aligned_cropped_image"])
                # merge_batch
                if batch_index not in merge_batch.keys():
                    merge_batch[batch_index] = {
                        "image": original_frames[batch_index],
                        "image_index": batch_index,
                        "faces_data": {}
                    }

        face_batch_input = torch.stack(face_batch_input, 0)

        swap_process_only_input_queue.put(face_batch_input, block=True)
        _swapped_faces = swap_process_only_output_queue.get(block=True)
        # _swapped_faces = fs_model(face_batch_input)
        _swapped_faces = fs_model.transform_inference_outputs(_swapped_faces)


        swapped_faces = []
        for swapped_face in _swapped_faces:
            _swapped_face = np.array(swapped_face, dtype=np.uint8)
            swapped_faces.append(_swapped_face)

        if use_segmentation_mask:
            seg_masks = []
            yolo_results = yolo_segment_image(img=swapped_faces, yolo_model=face_mask_model, return_original_image=False)
            for yolo_result in yolo_results:
                mask = yolo_result[1]
                mask[mask < 26] = 0
                mask[mask >= 26] = 255
                element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                # mask = cv2.dilate(mask, element)
                mask = cv2.erode(mask, element)
                # mask = np.clip(mask, 0, 255)
                seg_masks.append(mask)

        else:
            seg_masks = [np.ones(shape=(image_size, image_size, 1), dtype=np.uint8) * 255 for _ in range(len(swapped_faces))]



        index = 0
        for (batch_index, faces_data) in batch_extracted_new_params.items():
            if not faces_data:
                continue
            for (face_index, face_data) in faces_data.items():
                merge_batch[batch_index]["faces_data"][face_index] = \
                    (
                        swapped_faces[index],
                        face_data["aligned_cropped_image"],
                        face_data["aligned_cropped_params"],
                        seg_masks[index],
                    )
                index += 1

        print(
            f"inference [queue_index={queue_index}-b{batch_size}] total_inputs={len(batch_processed_face_swap_input_images)}) for face swapper took {(time.time() - swap_secs):.2f} secs")

        put_process_time = time.time()

        m_index = 0

        current_merge_queue_index = 0

        while m_index < batch_size:
            if m_index in merge_batch.keys():

                merge_input_queues[current_merge_queue_index].put((queue_index, merge_batch[m_index].copy(), batch_extrapolated_bboxes[m_index]), block=True)

                current_merge_queue_index += 1
                m_index += 1

                # reset the merge queue index
                if current_merge_queue_index >= len(merge_input_queues):
                    current_merge_queue_index = 0
            else:
                merge_output_queue.put((queue_index, m_index, original_frames[m_index].copy(), batch_extrapolated_bboxes[m_index], 0, 0), block=True)
                m_index += 1

        # del batch_processed_face_swap_input_images
        # print(f"put_process_time [queue_index={queue_index}] took {(time.time() - put_process_time):.2f} secs")
        print(f"total_inference_time [queue_index={queue_index}] took {(time.time() - total_secs):.2f} secs")
        if first_run:
            first_run = False

    print("swap process end reached")


def merge_faces_to_frame(input_queue, output_queue, seamless_clone, blur_mask):
    while True:
        queue_inputs = input_queue.get(block=True)
        if queue_inputs is None:
            output_queue.put(None, block=True)
            continue

        queue_index, merge_image_data, extrapolated_bboxes = queue_inputs
        merge_time = time.time()
        # {
        #     "image": image,
        #       "
        #     "faces_data": []
        # }
        image = merge_image_data["image"]
        image_index = merge_image_data["image_index"]

        for (face_index, face_data) in merge_image_data["faces_data"].items():

            swapped_image, aligned_cropped_image, aligned_cropped_params, seg_mask = face_data
            # swapped_image = color_transfer(swapped_image, aligned_cropped_image)

            # aligned_cropped_image = cv2.resize(aligned_cropped_image, (swapped_image.shape[1], swapped_image.shape[0]), interpolation=cv2.INTER_CUBIC)
            swapped_image_ct = color_transfer(swapped_image, aligned_cropped_image)

            smooth_average_passes = 2
            for _ in range(smooth_average_passes):
                swapped_image_ct = np.mean([swapped_image, swapped_image_ct], axis=0)

            swapped_image = swapped_image_ct

            # swapped_image = cv2.resize(swapped_image, (aligned_cropped_image.shape[1], aligned_cropped_image.shape[0]), interpolation=cv2.INTER_CUBIC)
            # seg_mask = cv2.resize(seg_mask, (swapped_image.shape[1], swapped_image.shape[0]),
            #                       interpolation=cv2.INTER_CUBIC)
            # swapped_image = colorTransfer(swapped_image, aligned_cropped_image, seg_mask)
            # swapped_image = blendImages(swapped_image, aligned_cropped_image, seg_mask)

            # seg_mask = np.ones_like(swapped_image, dtype=np.uint8)*255
            # seg_mask = seg_mask[..., 0]
            # swapped_image = unsharp_mask(swapped_image, amount=.5)


            image = paste_swapped_image(
                dst_image=image,
                swapped_image=swapped_image,
                seg_mask=seg_mask,
                aligned_cropped_params=aligned_cropped_params,
                seamless_clone=seamless_clone,
                blur_mask=blur_mask,
                resize=False
            )

            # array_to_img(seg_mask).save(
            #     f"C:/Users/teckt/PycharmProjects/iae_dfstudio/dfs_face_extractor/preds/face_extractor_preds_mask-{queue_index}-{image_index}.jpg",
            #     "JPEG")

            # array_to_img(rotated_image[..., :3]).save(
            #     f"C:/Users/teckt/PycharmProjects/iae_dfstudio/dfs_face_extractor/preds/face_extractor_preds-{set_num}-{image_index}-{int(face_index)}-3_re-rotated.jpg",
            #     "JPEG")

            # array_to_img(swapped_image).save(
            #     f"C:/Users/teckt/PycharmProjects/iae_dfstudio/dfs_face_extractor/preds/face_extractor_swapped-{set_num}-{batch_extracted_indexes[current_batch_index + i]}-{face_index}.jpg",
            #     "JPEG")
            # print(
            #     f"post processing outputs for face swapper [{queue_index}-{current_batch_index + i}] took {(time.time() - post_process_secs):.2f} secs")
            # print("before",image.shape)
            # image = draw_landmarks(image, np.expand_dims(extrapolated_bbox, axis=0))
            # print("after lm",image.shape)
            # image = draw_bbox(image / 255, np.expand_dims(extrapolated_bbox, axis=0))
            # image = np.array(image).astype(dtype=np.uint8)
            # print("after bbox",image.shape)
            # original_frames[image_index] = image
            # print(f"drawing outputs [{set_num}-{batch_size}] took {(time.time() - secs):.2f} secs")
            # array_to_img(image).save(
            #     f"C:/Users/teckt/PycharmProjects/iae_dfstudio/dfs_face_extractor/preds/face_extractor_preds_merged-{queue_index}-{image_index}-{face_index}.jpg",
            #     "JPEG")
        merge_time = time.time() - merge_time
        # if merge_time > 0.3:
        #     print(f"merge_time took {merge_time:.2f} secs, queue_index({queue_index}), image_index({image_index}), faces({len(merge_image_data['faces_data'])})")
        #     array_to_img(swapped_image).save(
        #         f"C:/Users/teckt/PycharmProjects/iae_dfstudio/dfs_face_extractor/preds/face_extractor_preds_swapped-{queue_index}-{image_index}-{merge_time:.2f}.jpg",
        #         "JPEG")
        # else:
        #     print(f"merge_time took {merge_time:.2f} secs")
        output_queue.put((queue_index, image_index, image.copy(), extrapolated_bboxes.copy(), len(merge_image_data["faces_data"]), merge_time), block=True)

def blendImages(src, dst, mask, featherAmount=0.2):
    #indeksy nie czarnych pikseli maski

    maskIndices = np.where(mask != 0)
    #te same indeksy tylko, ze teraz w jednej macierzy, gdzie kazdy wiersz to jeden piksel (x, y)
    maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    featherAmount = featherAmount * np.max(faceSize)

    maskPts = maskPts.astype(np.int32)
    hull = cv2.convexHull(maskPts)
    hull = hull.astype(np.int32)

    dists = np.zeros(maskPts.shape[0])
    for i in range(maskPts.shape[0]):
        dists[i] = cv2.pointPolygonTest(hull, (maskPts[i, 0].astype(np.uint8), maskPts[i, 1].astype(np.uint8)), True)

    weights = np.clip(dists / featherAmount, 0, 1)

    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

    return composedImg

#uwaga, tutaj src to obraz, z ktorego brany bedzie kolor
def colorTransfer(src, dst, mask):
    transferredDst = np.copy(dst)
    #indeksy nie czarnych pikseli maski
    maskIndices = np.where(mask != 0)
    #src[maskIndices[0], maskIndices[1]] zwraca piksele w nie czarnym obszarze maski

    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst

    return transferredDst

def prepare_single_image_for_face_extractor(input_file_path, max_frame_size):
    pil_img = Image.open(input_file_path)
    frame = np.array(pil_img).astype(dtype='uint8')

    h, w = frame.shape[:2]

    max_side = max(w, h)
    if max_side > max_frame_size:
        ratio = max_frame_size / max_side
        w = int(min(ratio * w, max_frame_size))
        h = int(min(ratio * h, max_frame_size))
    else:
        ratio = 1

    print("ratio", ratio, "w", w, "h", h)

    if ratio != 1:
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

    original_frames = [frame.copy()]

    preprocessed_frame = fit_to_size(frame, (416, 416), dtype='uint8', norm=False)
    pre_processed_inputs = np.array([preprocessed_frame])
    return original_frames, pre_processed_inputs

def prepare_for_swap_inference(fs_model, original_frames, batch_extracted_new_params):

    merge_batch = {}
    # put them all in a real batch
    face_batch_input = []
    mask_batch_input = []
    for (batch_index, faces_data) in batch_extracted_new_params.items():
        if not faces_data:
            continue
        for (face_index, face_data) in faces_data.items():
            # processed_image = np.clip(face_data["processed_image"] / 255., 0., 1.)
            processed_image = fs_model.transform_inference_inputs(face_data["processed_image"])
            # resize images if df

            face_batch_input.append(processed_image)
            mask_batch_input.append(face_data["aligned_cropped_image"])
            # merge_batch
            if batch_index not in merge_batch.keys():
                merge_batch[batch_index] = {
                    "image": original_frames[batch_index],
                    "image_index": batch_index,
                    "faces_data": {}
                }

    # mask_batch_input = np.array(mask_batch_input)
    return merge_batch, face_batch_input, mask_batch_input

def change_checkpoint(fs_model:FaceSwapModel, checkpoint_path):
    fs_model.load_checkpoint(checkpoint_path)
    # fs_model.load_lora_adapter(checkpoint_path+"-lora.pth")






