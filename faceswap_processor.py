import os.path
import time

import numpy as np
import torch
import cv2
import PIL.Image as Image
import numpy as np
import multiprocessing as mp
from multiprocessing.queues import Queue
import queue as queue_errors

from huggingface_hub import hf_hub_download
from insightface.utils import face_align
from torch import Tensor
import tqdm
from torchvision.transforms import transforms
from ultralytics import YOLO

from dfs_seg_inpaint_swap.faceswap.dfs_quick_swap.processes import swap_process_only
from dfs_seg_inpaint_swap.faceswap_model import denormalize_to_pil
from dfs_seg_inpaint_swap.tf_free_functions import align_crop_image, paste_swapped_image

face_mask_model_path = "yolov8s-face_mask.pt"
face_model_path = "yolov8n-face.pt"  # local path only


class FaceswapProcessor:
    def __init__(self, image_size, latent_dim, batch_size, align_processes_count, merge_processes_count):
        self.image_size = image_size

        self.video_path_queue = mp.Queue(maxsize=1)
        self.video_player_queue = mp.Queue(maxsize=0)  # be aware of the size when putting stuff in
        self.video_completed_queue = mp.Queue(maxsize=1)
        self.checkpoint_path = ""
        # self.checkpoint_queues = [mp.Queue(maxsize=1) for _ in range(align_processes_count)]
        # self.swap_process_only_input_queues = [torch.multiprocessing.Queue(maxsize=0) for _ in range(align_processes_count)]
        # self.swap_process_only_output_queues = [torch.multiprocessing.Queue(maxsize=0) for _ in range(align_processes_count)]
        # self.swap_process_only_processes = [mp.Process(
        #     target=swap_process_only,
        #     kwargs={
        #         "checkpoint_queue": self.checkpoint_queues[i],
        #         "swap_process_only_input_queue": self.swap_process_only_input_queues[i],
        #         "swap_process_only_output_queue": self.swap_process_only_output_queues[i],
        #     }
        # ) for i in range(align_processes_count)]
        # for p in self.swap_process_only_processes:
        #     p.start()

        self.checkpoint_queue = mp.Queue(maxsize=1)
        self.swap_process_only_input_queue = torch.multiprocessing.Queue(maxsize=0)

        self.swap_process_only_output_queue = torch.multiprocessing.Queue(maxsize=0)

        self.swap_process_only_process = mp.Process(
            target=swap_process_only,
            kwargs={
                "checkpoint_queue": self.checkpoint_queue,
                "swap_process_only_input_queue": self.swap_process_only_input_queue,
                "swap_process_only_output_queue": self.swap_process_only_output_queue,
            }
        )
        self.swap_process_only_process.start()
        # for p in self.swap_process_only_processes:
        #     p.start()

        self.merge_queues = [mp.Queue(maxsize=1) for _ in range(merge_processes_count)]
        self.merge_processes = [
            mp.Process(
                target=merge_process,
                kwargs={
                    "merge_queue": self.merge_queues[i],
                    "video_player_queue": self.video_player_queue,
                }
            ) for i in range(merge_processes_count)
        ]
        for p in self.merge_processes:
            p.start()

        # self.align_yolo_results_to_faceswap_tensors_queues = [mp.Queue(maxsize=2) for _ in range(align_processes_count)]
        # self.align_yolo_results_to_faceswap_tensors_processes = [
        #     mp.Process(
        #         target=align_yolo_results_to_faceswap_tensors,
        #         kwargs={
        #             "align_yolo_results_to_faceswap_tensors_queue": self.align_yolo_results_to_faceswap_tensors_queues[i],
        #             "faceswap_input_size": self.image_size,
        #             "faceswap_input_queue": self.swap_process_only_input_queues[i],
        #             "faceswap_output_queue": self.swap_process_only_output_queues[i],
        #             "merge_queues": self.merge_queues,
        #         }
        #     ) for i in range(align_processes_count)
        # ]
        # for p in self.align_yolo_results_to_faceswap_tensors_processes:
        #     p.start()

        self.face_swap_process = None

        self.video_player_process = None

        self.batch_size = batch_size

    def load_checkpoint(self, checkpoint_path):
        # # load the checkpoint
        # for checkpoint_queue in self.checkpoint_queues:
        #     checkpoint_queue.put(checkpoint_path)
        self.checkpoint_queue.put(checkpoint_path)
        self.checkpoint_path = checkpoint_path

    def swap_image(self, image):
        """
        swaps a single image from path or image
        :param image: (str|numpy array)
        :return: list[PIL.Image]
        """
        if isinstance(image, str):
            image = Image.open(image)
            image = np.array(image)
            # image = np.expand_dims(image, 0)
        preprocessed_image = self.model.transform_inference_inputs(image)
        preprocessed_image = preprocessed_image.unsqueeze(0)  # shape becomes (1, 3, h, w)
        outputs = self.model(preprocessed_image)

        return self.model.transform_inference_outputs(outputs)

    def swap_video(self, video_path, checkpoint, display_video, aligned_faces_dir=None):

        if self.checkpoint_path != checkpoint:
            self.load_checkpoint(checkpoint_path=checkpoint)

        self.face_swap_process = mp.Process(
            target=face_swap_app,
            kwargs={
                "video_path_queue": self.video_path_queue,
                "video_completed_queue": self.video_completed_queue,
                "faceswap_input_size": self.image_size,
                "faceswap_input_queue": self.swap_process_only_input_queue,
                "faceswap_output_queue": self.swap_process_only_output_queue,
                "merge_queues":self.merge_queues,
                "video_player_queue":self.video_player_queue,
                "max_faces": 4,
                "aligned_faces_dir": aligned_faces_dir
            }
        )

        self.start_video_player(video_path)

        self.face_swap_process.start()

        # start it
        self.video_path_queue.put(video_path, block=True)

    def start_video_player(self, video_path):

        if self.video_player_process is not None:
            self.video_player_queue.put(None, block=True)
            # while True:
            #     try:
            #         self.video_player_queue.get(block=False)
            #     except queue_errors.Empty:
            #         break

            self.video_player_process.join()
            print("video player process completed...")

        cap = cv2.VideoCapture(video_path)
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        cap.release()
        self.video_player_process = mp.Process(
            target=cv2_display_video,
            kwargs={
                "video_player_queue": self.video_player_queue,
                "fps": fps,
            }
        )
        self.video_player_process.start()


def swap_image(model, image):
    """
    swaps a single image from path or image
    :param image: (str|numpy array)
    :return: list[PIL.Image]
    """
    if isinstance(image, str):
        image = Image.open(image)
        image = np.array(image)
        # image = np.expand_dims(image, 0)
    preprocessed_image = model.transform_inference_inputs(image)
    preprocessed_image = preprocessed_image.unsqueeze(0)  # shape becomes (1, 3, h, w)
    outputs = model(preprocessed_image)

    return model.transform_inference_outputs(outputs)


def extract_video(extract_queue, swap_queue, batch_size):
    image_size = 224
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    # load detector and aligner
    insight_face_app = FaceAnalysis(name='buffalo_l')
    insight_face_app.prepare(ctx_id=0, det_size=(640, 640))

    from dfs_seg_inpaint_swap.faceswap_model import FaceSwapModel, FaceSwapModel2x
    # load face swap model
    image_size = 224
    latent_dim = 512
    fs_model = FaceSwapModel2x((3, image_size, image_size), dims=latent_dim)

    while True:

        video_path = extract_queue.get(block=True)

        if video_path is None:
            break
        cap = cv2.VideoCapture(video_path)
        batch = []
        batch_index = 0

        secs = time.time()
        while True:

            ret, frame = cap.read()
            if not ret:
                break

            faces = detect_align_face(insight_face_app, bgr_img=frame)
            np_aligned_cropped_rgb_faces = {}
            for i, face in enumerate(faces):
                aimg, M = face_align.norm_crop2(frame, face.kps, image_size)
                # cv2.imwrite(f"t1_face_{i}.jpg", aimg)
                rbg = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
                np_aligned_cropped_rgb_faces[i] = (rbg, M)

            batch.append((frame, np_aligned_cropped_rgb_faces))
            if len(batch) >= batch_size:
                # print(f"putting batch {batch_index} into swap_queue")
                with torch.no_grad():
                    tensors = stack_faces(batch, fs_model).pin_memory("cuda")
                print(f"extracting faces {time.time()-secs}")
                swap_queue.put((batch_index, batch, tensors), block=True)
                batch_index += 1
                batch = []
                secs = time.time()

        cap.release()
        print("swapping video finished")

    print("swap_video process finished")


def stack_faces(batch, fs_model):
    secs = time.time()
    prepared_batch = []
    # stack all faces into (B, H, W, C)
    for b, np_aligned_cropped_rgb_faces in batch:
        np_swapped_aligned_cropped_rgb_faces = {}
        for i, (face, M) in np_aligned_cropped_rgb_faces.items():
            face = fs_model.transform_inference_inputs(face)
            prepared_batch.append(face)

        #     swapped_pil_image: Image.Image = swap_image(fs_model, face)[0]
        #     # swapped_pil_image.show("Swapped Image")
        #     np_swapped_aligned_cropped_rgb_faces[i] = (np.array(swapped_pil_image), M)
        # swapped_batch.append(np_swapped_aligned_cropped_rgb_faces)

    print(f"preparing batch {(time.time() - secs):.2f}")

    secs = time.time()
    prepared_batch = torch.stack(prepared_batch, 0)
    print(f"stacking batch {(time.time() - secs):.2f}")
    return  prepared_batch


def stack_yolo_faces(results, fs_model):
    secs = time.time()
    prepared_batch = []
    # stack all faces into (B, H, W, C)
    for result in results:
        np_swapped_aligned_cropped_rgb_faces = {}
        for i, (face, M) in np_aligned_cropped_rgb_faces.items():
            face = fs_model.transform_inference_inputs(face)
            prepared_batch.append(face)

        #     swapped_pil_image: Image.Image = swap_image(fs_model, face)[0]
        #     # swapped_pil_image.show("Swapped Image")
        #     np_swapped_aligned_cropped_rgb_faces[i] = (np.array(swapped_pil_image), M)
        # swapped_batch.append(np_swapped_aligned_cropped_rgb_faces)

    print(f"preparing batch {(time.time() - secs):.2f}")

    secs = time.time()
    prepared_batch = torch.stack(prepared_batch, 0)
    print(f"stacking batch {(time.time() - secs):.2f}")
    return  prepared_batch


def unstack_faces(batch, swapped_batch_stacked):
    image_index = 0
    swapped_batch = []
    for b, np_aligned_cropped_rgb_faces in batch:
        np_swapped_aligned_cropped_rgb_faces = {}
        for i, (face, M) in np_aligned_cropped_rgb_faces.items():
            tensor:Tensor = swapped_batch_stacked[image_index]
            tensor = tensor.permute([1,2,0]) * 0.5 + 0.5
            tensor = torch.clip(tensor * 255, 0, 255)
            np_array = tensor.to("cpu").numpy()
            np_swapped_aligned_cropped_rgb_faces[i] = (np_array, M)

            image_index += 1

        #     swapped_pil_image: Image.Image = swap_image(fs_model, face)[0]
        #     # swapped_pil_image.show("Swapped Image")
        #     np_swapped_aligned_cropped_rgb_faces[i] = (np.array(swapped_pil_image), M)
        swapped_batch.append(np_swapped_aligned_cropped_rgb_faces)
    return swapped_batch


def swap_video(swap_queue, merge_queue):

    from dfs_seg_inpaint_swap.faceswap_model import FaceSwapModel
    # load face swap model
    image_size = 224
    latent_dim = 512
    checkpoint_path = f"faceswap/autoencoder_{image_size}_{latent_dim}.pth"

    fs_model = FaceSwapModel((3, image_size, image_size), dims=latent_dim)
    fs_model.load_checkpoint(checkpoint_path)
    fs_model.to("cuda")

    while True:
        batch_index, batch, stacked_faces = swap_queue.get(block=True)
        print(f"swap_video received batch {batch_index}")

        secs = time.time()
        with torch.no_grad():
            stacked_swapped_batch = fs_model.predict(stacked_faces.to("cuda"))
            print(f"swapping batch {(time.time() - secs):.2f}")

        merge_queue.put((batch_index, batch, stacked_swapped_batch.to("cpu")))


def merge_process(merge_queue, video_player_queue):
    while True:
        data, paste_data = merge_queue.get(block=True)
        # secs = time.time()
        swapped_frame = data['frame']
        if len(paste_data) == 0:
            video_player_queue.put((data["index"], swapped_frame))
            continue

        for _paste_data in paste_data:
            if use_norm_crop:
                swapped_frame = paste_face(swapped_frame, *_paste_data)
            else:
                swapped_frame = paste_swapped_image(swapped_frame, *_paste_data)

        video_player_queue.put((data["index"], swapped_frame))
        # print(f"[{data['index']}] sent to video player {(time.time() - secs):.2f}")
        continue

        swapped_batch = unstack_faces(batch, stacked_swapped_batch)
        print(f"unstacking batch {(time.time() - secs):.2f}")

        secs = time.time()
        for b, np_swapped_aligned_cropped_rgb_faces in enumerate(swapped_batch):
            # paste back
            np_aligned_cropped_rgb_faces = batch[b][1]
            img = batch[b][0]  # original image

            for i, (rgb_face, M) in np_swapped_aligned_cropped_rgb_faces.items():
                bgr_face = np_aligned_cropped_rgb_faces[i][0]
                M = np_aligned_cropped_rgb_faces[i][1]
                swapped_bgr_face = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2BGR)
                img = paste_face(img, bgr_face, swapped_bgr_face, M)

            # result_batch.append(img)
            video_player_queue.put(img)
        print(f"merging batch {(time.time() - secs):.2f}")


def detect_align_face(insight_face_app, bgr_img):
    """

    :param bgr_img:
    :return: list of Face, sorted by x_min (left most first)
    """

    faces = insight_face_app.get(bgr_img)
    faces = sorted(faces, key=lambda x: x.bbox[0])

    return faces


def cv2_display_video(video_player_queue, fps, max_res=640):
    import time
    frame_index = 0
    frames = {}
    exit_received = False
    frame_secs = {}
    frame_secsint = {}
    while True:
        received_frame = False
        try:
            # frame = video_player_queue.get(block=False)
            frame_data = video_player_queue.get(block=False)
            received_frame = True
        except queue_errors.Empty:
            pass
            # continue

        if received_frame:
            if frame_data is None:
                exit_received = True
                # if frame is None:
                # empty queue
                while True:
                    try:
                        video_player_queue.get(block=False)
                    except queue_errors.Empty:
                        break

                cv2.destroyAllWindows()
                break

            # cache frame if index doesn't match
            index, frame = frame_data
            frame_secs[index] = time.time()
            if index != frame_index:
                frames[index] = frame.copy()
                print(f"received current frame: {frame_index}, received frame: {index}")
                continue
        else:
            # check if current frame is in cache
            if frame_index in frames.keys():
                index, frame = frame_index, frames.pop(frame_index)
                print(f"cached current frame: {frame_index}, received frame: {index}")
            else:
                continue

        # max_side = max(frame.shape[:2])
        # if max_side > max_res:
        #     ratio = max_res / max_side
        #     new_w = int(min(ratio * frame.shape[1], max_res))
        #     new_h = int(min(ratio * frame.shape[0], max_res))
        #     # switched w,h because cv2 format
        #     frame = cv2.resize(frame, (new_w, new_h))
        if frame_index > 0:
            frame_secsint[frame_index] = time.time()-frame_secs[frame_index-1]
            frame_fps = len(frame_secsint) / sum(frame_secsint.values())
        else:
            frame_fps = 0

        # cv2.imshow('Result', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # cv2.imshow(f'Result', frame)
        frame_index += 1

        print(f"showing frame {frame_index} @ {frame_fps} fps'")
        # cv2.imwrite(f"faceswap/video/result_{str(frame_index).zfill(5)}.jpg", frame)
        # cv2.waitKey(int(1000 / fps))


def paste_face(original, aligned_bgr_face, swapped_aligned_bgr_face, M):
    target_img = original
    fake_diff = swapped_aligned_bgr_face.astype(np.float32) - aligned_bgr_face.astype(np.float32)
    fake_diff = np.abs(fake_diff).mean(axis=2)
    fake_diff[:2, :] = 0
    fake_diff[-2:, :] = 0
    fake_diff[:, :2] = 0
    fake_diff[:, -2:] = 0
    IM = cv2.invertAffineTransform(M)
    img_white = np.full((aligned_bgr_face.shape[0], aligned_bgr_face.shape[1]), 255, dtype=np.float32)
    bgr_fake = cv2.warpAffine(swapped_aligned_bgr_face, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    fake_diff = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    img_white[img_white > 20] = 255
    fthresh = 10
    fake_diff[fake_diff < fthresh] = 0
    fake_diff[fake_diff >= fthresh] = 255
    img_mask = img_white
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
    mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
    mask_size = int(np.sqrt(mask_h * mask_w))
    k = max(mask_size // 10, 10)
    # k = max(mask_size//20, 6)
    # k = 6
    kernel = np.ones((k, k), np.uint8)
    img_mask = cv2.erode(img_mask, kernel, iterations=1)
    kernel = np.ones((2, 2), np.uint8)
    fake_diff = cv2.dilate(fake_diff, kernel, iterations=1)
    k = max(mask_size // 20, 5)
    # k = 3
    # k = 3
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
    k = 5
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
    img_mask /= 255
    fake_diff /= 255
    # img_mask = fake_diff
    img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
    fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(np.float32)
    fake_merged = fake_merged.astype(np.uint8)
    return fake_merged


def align_yolo_results_to_faceswap_tensors(outputs_data, faceswap_input_queue, faceswap_output_queue, faceswap_input_size, merge_queues, video_player_queue, video_path, max_faces=4, landmarks_padding_ratio=1, debug_log=True, aligned_faces_dir=None):
    """

    :param outputs_data: (list[dicts]) a batch of yolo results in data interface
    :return:
    """

    # outputs_data = align_yolo_results_to_faceswap_tensors_queue.get(block=True)
    secs = time.time()

    for out_idx, data in enumerate(outputs_data):
        frame = data['frame']
        frame_index = data['index']
        # swapped_frame = frame.copy()
        keypointsn = data['keypointsn']
        keypoints = data['keypoints']
        boxes_data = data['boxes_data']
        aligned_cropped_images = {}
        aligned_swapped_images = {}
        aligned_cropped_params = {}
        # print(f"align swap copied frame", time.time() - secs)
        paste_datas = []
        if len(keypoints) > 0:

            confs = boxes_data[:, -2]
            for idx, xy in enumerate(keypoints):  # tensor of (ids,5,2)
                if idx >= len(confs):
                    continue
                if confs[idx] < 0.3:
                    continue

                if idx + 1 > max_faces:
                    break
                if debug_log:
                    print(f"align swap before align", time.time() - secs)

                if use_norm_crop:
                    aimg, M = face_align.norm_crop2(frame, xy, faceswap_input_size)
                    # cv2.imwrite(f"t1_face_{i}.jpg", aimg)
                    processed_image = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
                else:
                    try:
                        # xyn = keypointsn[idx]
                        # new_xy = xyn.copy()
                        # new_xy[..., 0] = new_xy[..., 0] * new_w
                        # new_xy[..., 1] = new_xy[..., 1] * new_h
                        aligned_cropped_image, aligned_cropped_param = align_crop_image(
                            original_image=frame,
                            original_landmarks=xy,
                            landmarks_padding_ratio=landmarks_padding_ratio
                        )
                    except:
                        if debug_log:
                            print(f"align swap after align error", time.time() - secs)
                        continue

                    # verify if out of bounds
                    if aligned_cropped_image.shape[0] >= max(frame.shape[:2]) or \
                            aligned_cropped_image.shape[1] >= max(frame.shape[:2]):
                        continue
                    try:
                        # save extracted faces here
                        if aligned_faces_dir is not None:
                            basename = os.path.basename(video_path)
                            cv2.imwrite(f"{aligned_faces_dir}/{basename}_{str(frame_index).zfill(5)}_{str(idx).zfill(1)}.jpg", aligned_cropped_image)

                        processed_image = cv2.resize(aligned_cropped_image, (faceswap_input_size, faceswap_input_size),
                                                     interpolation=cv2.INTER_CUBIC)
                        # swapped_image_bgr = processed_image
                        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                    except cv2.error:
                        continue

                if debug_log:
                    print(f"align swap after align", time.time() - secs)

                # send to face swap here
                tensor: Tensor = rgb_to_faceswap_tensor(processed_image)
                tensor = tensor.unsqueeze(0)
                faceswap_input_queue.put(tensor.pin_memory("cuda:0"), block=True)
                swapped_image_tensor = faceswap_output_queue.get(block=True)
                swapped_image_pil = denormalize_to_pil(swapped_image_tensor)[0]
                swapped_image_bgr = cv2.cvtColor(np.array(swapped_image_pil), cv2.COLOR_RGB2BGR)

                # swapped_image = unsharp_mask(swapped_image, amount=.75)
                print(f"align swap after swap", time.time() - secs)

                if use_norm_crop:
                    paste_data = (aimg, swapped_image_bgr, M)
                    paste_datas.append(paste_data)

                else:
                    seg_mask = np.ones(shape=(faceswap_input_size, faceswap_input_size, 1), dtype=np.uint8) * 255
                    paste_data = (swapped_image_bgr, seg_mask, aligned_cropped_param, True, False, False, True, 1)
                    paste_datas.append(paste_data)
                #
                # aligned_cropped_images[idx] = aligned_cropped_image
                # aligned_cropped_params[idx] = aligned_cropped_param
                # aligned_swapped_images[idx] = swapped_image_bgr

                # data.update({
                #     'swapped_frame': paste_face(frame, aimg, swapped_image_bgr, M),
                #     'aligned_cropped_images': aligned_cropped_images,
                #     'aligned_cropped_params': aligned_cropped_params,
                #     'aligned_swapped_images': aligned_swapped_images,
                # })
                # print(f"align swap after merge", time.time() - secs)
        # data, paste_data = merge_queue.get(block=True)
        # secs = time.time()
        # swapped_frame = data['frame']
        # if len(paste_datas) == 0:
        #     video_player_queue.put((data["index"], swapped_frame))
        #     continue
        #
        # for _paste_data in paste_datas:
        #     if use_norm_crop:
        #         swapped_frame = paste_face(swapped_frame, *_paste_data)
        #     else:
        #         swapped_frame = paste_swapped_image(swapped_frame, *_paste_data)
        #
        # video_player_queue.put((data["index"], swapped_frame))
        is_put_in = False
        while True:
            if is_put_in:
                break
            for merge_queue in merge_queues:
                try:
                    merge_queue.put((data, paste_datas), block=False)
                    is_put_in = True
                    break
                except queue_errors.Full:
                    continue
        # data.update({
        #         'swapped_frame': swapped_frame,
        #         'aligned_cropped_images': aligned_cropped_images,
        #         'aligned_cropped_params': aligned_cropped_params,
        #         'aligned_swapped_images': aligned_swapped_images,
        #     })

    # print(f"align swap", time.time() - secs)


def face_swap_app(video_path_queue, video_completed_queue, faceswap_input_size, faceswap_input_queue, faceswap_output_queue, merge_queues, video_player_queue, max_faces=4, landmarks_padding_ratio=1, aligned_faces_dir="aligned"):

    face_extractor = YOLO(hf_hub_download("Anyfusion/xseg", "yolov8n-face.pt"))

    while True:
        video_path = video_path_queue.get(block=True)
        if video_path is None:
            break

        yolo_gen = face_extractor(video_path, stream=True, vid_stride=30)

        outputs_data = []
        frame_index = 0
        for result in tqdm.tqdm(yolo_gen):
            keypoints = result.keypoints.xy.cpu().numpy()
            keypointsn = result.keypoints.xyn.cpu().numpy()
            boxes_data = result.boxes.data.cpu().numpy()
            data = {
                "index": frame_index,
                "frame": result.orig_img,
                "keypoints": keypoints,
                "boxes_data": boxes_data,
                "keypointsn": keypointsn
            }
            outputs_data.append(data)

            if len(outputs_data) == 4:
                # is_put_in = False
                # while True:
                #     if is_put_in:
                #         break
                #     for align_yolo_results_to_faceswap_tensors_queue in align_yolo_results_to_faceswap_tensors_queues:
                #         try:
                #             align_yolo_results_to_faceswap_tensors_queue.put(outputs_data, block=False)
                #             is_put_in = True
                #             break
                #         except queue_errors.Full:
                #             continue

                align_yolo_results_to_faceswap_tensors(
                    outputs_data,
                    faceswap_input_queue=faceswap_input_queue,
                    faceswap_output_queue=faceswap_output_queue,
                    faceswap_input_size=faceswap_input_size,
                    merge_queues=merge_queues,
                    video_player_queue=video_player_queue,
                    max_faces=max_faces,
                    landmarks_padding_ratio=landmarks_padding_ratio,
                    video_path=video_path,
                    aligned_faces_dir=aligned_faces_dir,
                )
                # for data in outputs_data:
                #     video_player_queue.put(data['swapped_frame'], block=True)

                outputs_data = []

            frame_index += 1
        print("video finished streaming")
        video_player_queue.put(None, block=True)
        video_completed_queue.put(True, block=True)



def rgb_to_faceswap_tensor(rgb):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    return transform(rgb)

use_norm_crop = False
if __name__ == "__main__":
    image_size = 224
    latent_dim = 512
    checkpoint = f"faceswap/autoencoder_{image_size}_{latent_dim}.pth"

    fsp = FaceswapProcessor(
        image_size=image_size, latent_dim=latent_dim, batch_size=4,
        align_processes_count=2, merge_processes_count=2
    )
    vid_root = "C:/Users/teckt/Documents/test_vids/F"
    for f in os.listdir(vid_root):
        vid_path = f"{vid_root}/{f}"
        if os.path.isfile(vid_path):
            fsp.swap_video(
                video_path=vid_path,
                aligned_faces_dir="C:/Users/teckt/Documents/test_vids/F/aligned",
                checkpoint=checkpoint, display_video=True)
            fsp.video_completed_queue.get(block=True)
