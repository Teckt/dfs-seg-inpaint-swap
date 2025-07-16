import multiprocessing as mp
import os
import subprocess
import time
import queue as queue_errors
from math import floor
import PIL.Image as Image

import numpy as np
import cv2
import torch

from dfs_seg_inpaint_swap.faceswap.processes import read_frames_into_queue, extract_faces, merge_faces_to_frame, \
    swap_faces_multi_face_fix, prepare_single_image_for_face_extractor, swap_process_only, detect_process_only


xseg_model_path = "C:/Users/teckt/PycharmProjects/iae_dfstudio/models/xseg/saved_model"

max_faces = 4
landmarks_padding_ratio = 1
conf_threshold = 0.40


def cv2_write_video(input_queue, source_video_file, output_video_file, fps, fourcc, use_watermark, max_frame_size, use_shortest, upscale_resolution_hw=None):

    videoThumbnail = output_video_file.replace(os.path.basename(output_video_file), "videoThumbnail.jpg")

    frame_index = 0
    output_video = None
    while True:
        frame = input_queue.get(block=True)
        if frame is None:
            # empty queue
            while True:
                try:
                    input_queue.get(block=False)
                except queue_errors.Empty:
                    break

            output_video.release()
            tmp_copy = output_video_file + "_copy"
            os.rename(output_video_file, tmp_copy)

            shortest_ = '-shortest' if use_shortest else ""

            cmd = 'ffmpeg -y' \
                  ' -i ' + tmp_copy + \
                  ' -i "' + source_video_file + '" -c copy -map 0:v:0 -map 1:a?:0 '+shortest_+' "' + output_video_file + '"'
            subprocess.run(cmd)
            os.remove(tmp_copy)

            break

        h,w = frame.shape[:2]

        # max_side = max(w, h)
        # if use_watermark:
        #     watermark_frame(frame)

        if upscale_resolution_hw is not None:
            upscale_h, upscale_w = upscale_resolution_hw
            frame = cv2.resize(frame, (upscale_w, upscale_h), interpolation=cv2.INTER_CUBIC)



        # if max_side > max_frame_size:
        #     ratio = max_frame_size / max_side
        #     w = int(min(ratio * w, max_frame_size))
        #     h = int(min(ratio * h, max_frame_size))
        #     frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)

        # print("ratio", ratio, "w", w, "h", h)


        stream_resolution = (frame.shape[1], frame.shape[0])

        if output_video is None:
            output_video = cv2.VideoWriter(output_video_file, fourcc, fps, stream_resolution)

        output_video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # save thumbnail to send to user as swap preview
        if frame_index % fps == 0:
            max_res = 480
            max_side = max(frame.shape[:2])
            if max_side > max_res:
                ratio = max_res / max_side
                new_w = int(min(ratio * frame.shape[1], max_res))
                new_h = int(min(ratio * frame.shape[0], max_res))
                # switched w,h because cv2 format
                frame = cv2.resize(frame, (new_w, new_h))
            Image.fromarray(frame).save(videoThumbnail)

        frame_index += 1


def cv2_display_video(input_queue, fps, max_res=640):
    while True:
        frame = input_queue.get(block=True)
        if frame is None:
            # empty queue
            while True:
                try:
                    input_queue.get(block=False)
                except queue_errors.Empty:
                    break

            cv2.destroyAllWindows()
            break

        max_side = max(frame.shape[:2])
        if max_side > max_res:
            ratio = max_res / max_side
            new_w = int(min(ratio * frame.shape[1], max_res))
            new_h = int(min(ratio * frame.shape[0], max_res))
            # switched w,h because cv2 format
            frame = cv2.resize(frame, (new_w, new_h))

        cv2.imshow('Result', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(int(1000 / fps))


class VideoSwapper:
    def set_fps(self):
        cap = cv2.VideoCapture(self.input_video_file)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # if fps > 24:
        #     self.fps = 24
        # else:
        self.fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.num_frames = min(total_frames, self.fps*60*10)  # 10 minutes of frames
        self.original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"fps={self.fps}, frames={total_frames}, frames_to_swap={self.num_frames},  original_width={ self.original_width}, original_height={self.original_height}")
        cap.release()

    def __init__(self, batch_size=32, max_frame_size=720, processes_count=4, use_og_extractor=False, seamless_clone=True, blur_mask=True, display_stream=False, face_extract_process_count=2):
        super().__init__()

        self.use_og_extractor = use_og_extractor
        self.draw_bboxes_state = False

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
        self.stream_resolution = None
        self.fps = 0
        self.original_width = 0
        self.original_height = 0
        self.num_frames = 0
        self.output_video = None
        self.output_video_file = ""
        self.input_video_file = ""

        self.current_model_name = "jjk-enhanced-xM-noenc-syc"
        self.model_size = None

        self.display_stream = display_stream

        self.max_frame_size = max_frame_size
        self.processes_count = processes_count
        self.face_extract_process_count = face_extract_process_count
        self.batch_size = batch_size

        self.seamless_clone = seamless_clone
        self.blur_mask = blur_mask

        # cv2 reads video frames and preprocesses into a batch ready for face_extractor inference
        self.face_extractor_input_process_stop_signal_queue = mp.Queue(maxsize=1)
        self.face_extractor_input_queue = torch.multiprocessing.Queue(maxsize=4)

        self.batch_face_swap_input_queue = torch.multiprocessing.Queue(maxsize=4)

        self.merge_input_queues = [torch.multiprocessing.Queue(maxsize=0) for _ in range(self.processes_count)]
        self.merge_output_queue = torch.multiprocessing.Queue(maxsize=0)

        self.checkpoint_queue = mp.Queue(maxsize=1)
        self.swap_process_only_input_queue = torch.multiprocessing.Queue(maxsize=1)
        self.swap_process_only_output_queue = torch.multiprocessing.Queue(maxsize=1)
        self.swap_process_only = mp.Process(
            target=swap_process_only,
            kwargs={
                "checkpoint_queue": self.checkpoint_queue,
                "swap_process_only_input_queue": self.swap_process_only_input_queue,
                "swap_process_only_output_queue": self.swap_process_only_output_queue,
            }
        )
        self.swap_process_only.start()

        self.detect_process_only_input_queue = torch.multiprocessing.Queue(maxsize=1)
        self.detect_process_only_output_queue = torch.multiprocessing.Queue(maxsize=1)
        self.detect_process_only = mp.Process(
            target=detect_process_only,
            kwargs={
                "detect_process_only_input_queue": self.detect_process_only_input_queue,
                "detect_process_only_output_queue": self.detect_process_only_output_queue,
            }
        )
        self.detect_process_only.start()

        self.face_extractor_processes = [
            mp.Process(
                target=extract_faces,
                kwargs={
                    "input_queue": self.face_extractor_input_queue,

                    # "output_queue": self.postprocess_face_extractor_outputs_queue,
                    "output_queue": self.batch_face_swap_input_queue,
                    # "output_queue": self.postprocess_swapped_faces_queue,
                    "merge_output_queue": self.merge_output_queue,
                    # a merge queue manager that filters and puts them in order
                    "max_faces": max_faces,
                    "conf_threshold": conf_threshold,
                    "landmarks_padding_ratio": landmarks_padding_ratio,
                    "detect_process_only_input_queue": self.detect_process_only_input_queue,
                    "detect_process_only_output_queue": self.detect_process_only_output_queue

                }
            ) for _ in range(self.face_extract_process_count)
        ]

        self.face_extractor_process = mp.Process(
            target=extract_faces,
            kwargs={
                "input_queue": self.face_extractor_input_queue,

                # "output_queue": self.postprocess_face_extractor_outputs_queue,
                "output_queue": self.batch_face_swap_input_queue,
                # "output_queue": self.postprocess_swapped_faces_queue,
                "merge_output_queue": self.merge_output_queue,  # a merge queue manager that filters and puts them in order
                "max_faces": max_faces,
                "conf_threshold": conf_threshold,
                "landmarks_padding_ratio": landmarks_padding_ratio,

            })

        self.set_segmentation_mask_queue = mp.Queue(maxsize=1)

        # begin swap process

        self.swap_process = mp.Process(
            target=swap_faces_multi_face_fix,
            kwargs={
                "input_queue": self.batch_face_swap_input_queue,
                "merge_input_queues": self.merge_input_queues,
                "merge_output_queue": self.merge_output_queue,
                "set_segmentation_mask_queue": self.set_segmentation_mask_queue,
                "swap_process_only_input_queue": self.swap_process_only_input_queue,
                "swap_process_only_output_queue": self.swap_process_only_output_queue,
            })
        print("starting swap_process")
        self.swap_process.start()

        self.merge_processes = [
            mp.Process(
                target=merge_faces_to_frame,
                kwargs={
                    "input_queue": self.merge_input_queues[p],
                    "output_queue": self.merge_output_queue,
                    "seamless_clone": self.seamless_clone,
                    "blur_mask": self.blur_mask
                }
            ) for p in range(self.processes_count)]

        self.video_writer_queue = torch.multiprocessing.Queue(maxsize=0)  # be aware of the size when putting stuff in
        self.video_player_queue = torch.multiprocessing.Queue(maxsize=0)  # be aware of the size when putting stuff in

        self.merge_log_queue = mp.Queue(maxsize=0)  # be aware of the size when putting stuff in

        self.swap_process = None

        self.start()

    def set_checkpoint(self, checkpoint_path):
        self.checkpoint_queue.put(checkpoint_path)

    def set_use_segmentation_mask(self, use_segmentation_mask):
        self.set_segmentation_mask_queue.put(use_segmentation_mask)  # True == use as model_dir instead of model_name for model manager

    def toggle_draw_bboxes(self):
        self.draw_bboxes_state = not self.draw_bboxes_state
        print("Draw boxes =", self.draw_bboxes_state)

    def swap_image(self, checkpoint_path, input_file, output_file, max_frame_size, use_watermark):

        print("clearing the checkpoint")
        # clear the checkpoints if no face detected from extractor
        while self.checkpoint_queue.qsize() > 0:
            try:
                self.checkpoint_queue.get(block=False)
            except queue_errors.Empty:
                pass

        while not self.queues_are_empty():
            print("queues are not empty, sleeping 3...")
            time.sleep(3)

        print("setting the checkpoint")
        self.set_checkpoint(checkpoint_path)

        self.max_frame_size = max_frame_size

        # prepare the image
        print(f"preparing image for face extractor input from {input_file}")
        original_frames, pre_processed_inputs = prepare_single_image_for_face_extractor(input_file, max_frame_size)
        # send to image extract queue
        print(f"sending data to face extractor image_extract_input_queue")
        self.image_extract_input_queue.put((original_frames, pre_processed_inputs), block=True)
        # wait for image_output_queue for result
        print("awaiting image swap result")
        image = self.image_output_queue.get(block=True)
        h, w = image.shape[:2]

        max_side = max(w, h)
        # if use_watermark:
        #     watermark_frame(image)
        if max_side > max_frame_size:
            ratio = max_frame_size / max_side
            w = int(min(ratio * w, max_frame_size))
            h = int(min(ratio * h, max_frame_size))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)

        # print("ratio", ratio, "w", w, "h", h)

        Image.fromarray(image).save(output_file, 'JPEG', quality=95, optimize=True, progressive=True)

    def kill_swap_process(self):
        print("clearing the checkpoint")
        # clear the checkpoints if no face detected from extractor
        while self.checkpoint_queue.qsize() > 0:
            try:
                self.checkpoint_queue.get(block=False)
            except queue_errors.Empty:
                pass
        print("setting the checkpoint")
        self.checkpoint_queue.put(None, block=True)
        self.batch_face_swap_input_queue.put(None, block=True)
        self.swap_process.join()
        self.swap_process.kill()

    def swap_video(self, checkpoint_path, input_video_file, output_video_file, use_segmentation_mask, seamless_clone, blur_mask, max_frame_size, use_watermark, encode_shortest, upscale_output, secs_to_stop_at):
        # check if queues are empty before starting
        print("clearing the checkpoint")
        # clear the checkpoints if no face detected from extractor
        while self.checkpoint_queue.qsize() > 0:
            try:
                self.checkpoint_queue.get(block=False)
            except queue_errors.Empty:
                pass

        while not self.queues_are_empty():
            print("queues are not empty, sleeping 3...")
            time.sleep(3)

        # clear the merged dicts
        self.current_queue_index = 0
        self.frame_index = -1
        self.merged_frames_dict = {}

        print("setting the checkpoint")
        self.set_checkpoint(checkpoint_path)
        print("setting the segmentation mask flag")
        self.set_use_segmentation_mask(use_segmentation_mask)

        self.max_frame_size = max_frame_size
        print("setting the merged_output_process")
        self.merged_output_process = mp.Process(
            target=merged_output_processor,
            kwargs={
                "batch_size": self.batch_size,
                "merge_output_queue": self.merge_output_queue,
                "merge_log_queue": self.merge_log_queue,
                "video_writer_queue": self.video_writer_queue,
                "video_player_queue": self.video_player_queue if self.display_stream else None
            })

        self.input_video_file = input_video_file
        self.output_video_file = output_video_file

        self.set_fps()

        print("setting the face_extractor_input_process")
        if upscale_output and max(self.original_width, self.original_height) > 1920:
            # get the smallest from the original and max frame size
            self.max_frame_size = min(1920, max(self.original_width, self.original_height))
            print(f"downscaling input, self.original_width={self.original_width}, self.original_height={self.original_height} , self.max_frame_size={self.max_frame_size}")

        self.face_extractor_input_process = mp.Process(
            target=read_frames_into_queue,
            kwargs={
                "video_file_path": self.input_video_file,
                "stop_signal_queue": self.face_extractor_input_process_stop_signal_queue,
                "output_queue": self.face_extractor_input_queue,
                "batch_size": self.batch_size,
                # downscale the output so that we can upscale the output back to original_size
                "max_frame_size": self.max_frame_size,
                "exit_on_complete": True,
                "secs_to_stop_at": secs_to_stop_at
            })

        print("setting the video_writer_process")
        self.video_writer_process = mp.Process(
            target=cv2_write_video,
            kwargs={
                "input_queue": self.video_writer_queue,
                "source_video_file": self.input_video_file,
                "output_video_file": self.output_video_file,
                "fps": self.fps,
                "fourcc": self.fourcc,
                "max_frame_size": self.max_frame_size,
                "use_watermark": use_watermark,
                "use_shortest": encode_shortest,
                "upscale_resolution_hw": (self.original_height, self.original_width) if upscale_output else None,
                # deprecated in 2.2.4: the length of videos will always be the same for jobs
                # use True
            }
        )

        print("starting video_writer_process")
        self.video_writer_process.start()

        if self.display_stream:
            print("setting the video_player_process")
            self.video_player_process = mp.Process(
                target=cv2_display_video,
                kwargs={
                    "input_queue": self.video_player_queue,
                    "fps": self.fps,
                }
            )
            print("starting video_player_process")
            self.video_player_process.start()

        print("starting merged_output_process")
        self.merged_output_process.start()

        print("starting face_extractor_input_process")
        self.face_extractor_input_process.start()

    def stop_swap_video(self):

        # stop the display function to stop grabbing frames from the merged frames queue and putting frames in the video queues
        print("waiting for merged_output_process to complete...")
        self.merged_output_process.join()
        # self.merged_output_process.kill()
        # process auto stops after video complete, just join
        # self.face_extractor_input_process_stop_signal_queue.put(None, block=True)
        print("waiting for face_extractor_input_process to complete...")
        self.face_extractor_input_process.join()
        # self.face_extractor_input_process.kill()

        # stop swap process
        # while True:
        #     try:
        #         # print("self.batch_face_swap_input_queue", self.batch_face_swap_input_queue.qsize())
        #         self.batch_face_swap_input_queue.get(block=True, timeout=1)
        #     except queue_errors.Empty:
        #         break
        # self.batch_face_swap_input_queue.put(None, block=True)


        # self.queues_are_empty()

        # print("waiting for swap_process to complete...")
        # self.swap_process.kill()
        # self.swap_process.join()

        # stop the video processes by sending None to empty queues
        print("waiting for video writer processes to complete...")
        while self.video_writer_queue.qsize() > 0:
            time.sleep(1)
        self.video_writer_queue.put(None, block=True)
        self.video_writer_process.join()
        print("video writer process completed...")

        if self.display_stream:
            if self.video_player_process.is_alive():
                print("waiting for video player processes to complete...")
                while self.video_player_queue.qsize() > 0:
                    time.sleep(1)
                self.video_player_queue.put(None, block=True)
                self.video_player_process.join()
                print("video player process completed...")
        # empty queue
        while True:
            try:
                print("self.merge_output_queue", self.merge_output_queue.qsize())
                self.merge_output_queue.get(block=True, timeout=1)
            except queue_errors.Empty:
                if self.queues_are_empty():
                    break

        while True:
            try:
                print("self.merge_log_queue", self.merge_log_queue.qsize())
                self.merge_log_queue.get(block=False)
            except queue_errors.Empty:
                break

        while True:
            queues_emptied = self.queues_are_empty()
            print(f"queue emptied={queues_emptied}")
            if queues_emptied:
                break
            time.sleep(5)

    def queues_are_empty(self):
        if self.face_extractor_input_process_stop_signal_queue.qsize() != 0:
            print("face_extractor_input_process_stop_signal_queue", self.face_extractor_input_process_stop_signal_queue.qsize())
            return False

        if self.face_extractor_input_queue.qsize() != 0:
            print("face_extractor_input_queue", self.face_extractor_input_queue.qsize())
            return False

        if self.batch_face_swap_input_queue.qsize() != 0:
            print("batch_face_swap_input_queue", self.batch_face_swap_input_queue.qsize())
            return False

        for merge_input_queue in self.merge_input_queues:
            if merge_input_queue.qsize() != 0:
                print("merge_input_queue", merge_input_queue.qsize())
                return False

        if self.merge_output_queue.qsize() != 0:
            print("merge_output_queue", self.merge_output_queue.qsize())
            return False

        if self.video_writer_queue.qsize() != 0:
            print("video_writer_queue", self.video_writer_queue.qsize())
            return False

        if self.video_player_queue.qsize() != 0:
            print("video_player_queue", self.video_player_queue.qsize())
            return False

        if self.set_segmentation_mask_queue.qsize() != 0:
            print("set_segmentation_mask_queue", self.set_segmentation_mask_queue.qsize())
            return False

        if self.checkpoint_queue.qsize() != 0:
            print("checkpoint_queue", self.checkpoint_queue.qsize())
            return False

        return True

    def start(self):
        # self.face_extractor_process.start()
        for p in range(self.face_extract_process_count):
            self.face_extractor_processes[p].start()
        # self.swap_process.start()
        for p in range(self.processes_count):
            self.merge_processes[p].start()

def merged_output_processor(batch_size, merge_output_queue, merge_log_queue, video_writer_queue, video_player_queue=None, num_merge_processes=2, log=False):
    current_queue_index = 0
    merged_frames_dict = {}
    frame_index = -1

    nones_received = 0

    max_queue_index = 0

    last_batch_time = time.time()
    print(f"merged_output_processor started, num_merge_processes={num_merge_processes}")
    while True:
        if current_queue_index not in merged_frames_dict.keys():
            merged_frames_dict[current_queue_index] = {}

        if frame_index == -1:
            while True:
                try:
                    queue_inputs = merge_output_queue.get(block=False)
                except queue_errors.Empty:
                    break

                if queue_inputs is None:
                    nones_received += 1
                    print(f"received None @ max_queue_index={max_queue_index}, sleeping 2 seconds...")
                    time.sleep(2)
                    if nones_received >= num_merge_processes:
                        print(f"received {num_merge_processes} Nones @ max_queue_index={max_queue_index}")
                        # break
                    continue

                queue_index, image_index, merged_image, extrapolated_bbox, num_faces, merge_time = queue_inputs
                if queue_index > max_queue_index:
                    max_queue_index = queue_index
                if queue_index not in merged_frames_dict.keys():
                    merged_frames_dict[queue_index] = {}

                merged_frames_dict[queue_index][image_index] = {
                    "frame": merged_image,
                    "bbox": extrapolated_bbox,
                    "num_faces": num_faces,
                    "merge_time": min(max(merge_time, .005), 2),  # clip to these values
                }

                if log:
                    print(f"received frame, queue_index={queue_index}, image_index={image_index}, current_queue_index={current_queue_index}, frames_current_queue_index={len(merged_frames_dict[current_queue_index])}, max_queue_index={max_queue_index}, nones received={nones_received}")

            if nones_received >= num_merge_processes:
                print(f"{nones_received} nones_received, max_queue_index={max_queue_index}, looping current_queue_index={current_queue_index}, size of batch={len(merged_frames_dict[current_queue_index])}")

            if len(merged_frames_dict[current_queue_index]) == batch_size or (nones_received >= num_merge_processes and current_queue_index>=max_queue_index-1):
                # process the batch
                frame_index = 0
                batch_time = (time.time() - last_batch_time)

                total_merge_time = 0
                total_faces = 0
                for image_index, frame_info_dict in merged_frames_dict[current_queue_index].items():
                    total_merge_time += frame_info_dict["merge_time"]
                    total_faces += frame_info_dict["num_faces"]

                if total_faces == 0 or batch_time == 0:
                    frame_info_text = f"received batch {current_queue_index}(size={batch_size}) in {batch_time:.02f}, swap fps = {(batch_size / 0.5)}" + \
                                      f"\ntotal_faces={total_faces}, total_merge_time={total_merge_time:.02f}, per face={total_merge_time}"
                else:
                    frame_info_text = f"received batch {current_queue_index}(size={batch_size}) in {batch_time:.02f}, swap fps = {(batch_size / batch_time):.02f}" + \
                                      f"\ntotal_faces={total_faces}, total_merge_time={total_merge_time:.02f}, per face={(total_merge_time if total_faces == 0 else (total_merge_time / total_faces)):.02f}"

                if batch_time == 0:
                    merge_log_queue.put((current_queue_index, (batch_size / 0.5),), block=True)
                else:
                    merge_log_queue.put((current_queue_index, (batch_size / batch_time),), block=True)
                if log:
                    print(frame_info_text)
            continue

        else:
            if nones_received >= num_merge_processes and current_queue_index >= max_queue_index-1:
                print(f"merged_output_processor ended, current_queue_index={current_queue_index}, max_queue_index={max_queue_index}", )
                print("merge_output_queue", merge_output_queue.qsize())
                print("merge_log_queue", merge_log_queue.qsize())
                print("video_writer_queue", video_writer_queue.qsize())
                # print("video_player_queue", video_player_queue.qsize())
                # merge_output_queue, merge_log_queue, video_writer_queue, video_player_queue

                exit()
            else:

                frame = merged_frames_dict[current_queue_index][frame_index]["frame"]

                try:
                    video_writer_queue.put(frame)
                except queue_errors.Full:
                    pass

                if video_player_queue is not None:
                    try:
                        video_player_queue.put(frame)
                    except queue_errors.Full:
                        pass

                if log:
                    print(f"video writer/player frame sent, current_queue_index={current_queue_index} (frame_index={frame_index}, len(merged_frames_dict[current_queue_index])={len(merged_frames_dict[current_queue_index])})")

                frame_index += 1
                if frame_index == batch_size or frame_index == len(merged_frames_dict[current_queue_index]):
                    if nones_received >= num_merge_processes and current_queue_index >= max_queue_index-1:
                        print("merged_output_processor ended")
                        print("merge_output_queue", merge_output_queue.qsize())
                        print("merge_log_queue", merge_log_queue.qsize())
                        print("video_writer_queue", video_writer_queue.qsize())
                        # print("video_player_queue", video_player_queue.qsize())
                        exit()

                    del merged_frames_dict[current_queue_index]
                    frame_index = -1
                    current_queue_index += 1
                    last_batch_time = time.time()

            continue


def format_seconds(seconds):

    estimatedHours = floor(seconds / 60.0 / 60.0)

    estimatedMinutes = floor((seconds - estimatedHours * 60 * 60) / 60.0)

    estimatedRemainderSeconds = seconds - (estimatedMinutes * 60 + estimatedHours * 60 * 60)
    return f"{str(estimatedHours).zfill(2)}:{str(estimatedMinutes).zfill(2)}:{str(estimatedRemainderSeconds).zfill(2)}"


def swap_file(swapper, checkpoint_path, video_path, output_path, secs_to_stop_at):

    swapper.swap_video(
        checkpoint_path=checkpoint_path,
        input_video_file=video_path,
        output_video_file=output_path,
        use_segmentation_mask=True,
        seamless_clone=False,  # don't matter
        blur_mask=True,  # don't matter
        max_frame_size=2048,
        use_watermark=False,
        encode_shortest=True,
        upscale_output=False,
        secs_to_stop_at=secs_to_stop_at,
    )
    swap_fps_list = []

    _batch_size = swapper.batch_size
    _video_fps = swapper.fps
    _num_frames = swapper.num_frames

    while swapper.merged_output_process.is_alive():
        # TODO get the current merged frame index or however to get the progress and estimate
        # swap_fps, current_frame_index or batch_index, total_frames

        time_elapsed = 0
        wait_time_seconds = 1
        try:
            _queue_index, _swap_fps = swapper.merge_log_queue.get(block=False)
        except queue_errors.Empty:
            print(f"\r[{time_elapsed}]waiting for log...", end="")
            time.sleep(wait_time_seconds)
            time_elapsed += wait_time_seconds
            continue

        swap_fps_list.append(_swap_fps if _swap_fps < 50 else 1)
        avg_swap_fps = np.average(np.array(swap_fps_list))
        frames_swapped = (_batch_size * (_queue_index + 1))
        frames_remaining = _num_frames - frames_swapped
        time_remaining = frames_remaining / avg_swap_fps

        progress = int(1.0 * (_batch_size * (_queue_index + 1)) / _num_frames * 100)

        _log = f"Swapping {_batch_size * (_queue_index + 1)}/{_num_frames} - {_swap_fps:.02f}/s - {format_seconds(time_remaining)}"

        print(_log)

    print("finished!!")
    swapper.stop_swap_video()


def swap_dir(swapper, checkpoint_path, dir_path, out_dir, replace_existing, secs_to_stop_at):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    files = [f for f in os.listdir(dir_path) if f.endswith(".mp4") or f.endswith(".webm") or f.endswith(".ts")]

    existing_output_files = [f for f in os.listdir(dir_path) if f.endswith(".mp4") or f.endswith(".ts")]

    for file in files:
        if "swapped" in file:
            continue

        video_path = f"{dir_path}/{file}"
        output_path = f"{out_dir}/{file}_swapped.mp4"
        if not replace_existing:
            if os.path.exists(output_path):
                continue
        swap_file(swapper, checkpoint_path, video_path, output_path, secs_to_stop_at)


if __name__ == "__main__":
    swapper = VideoSwapper(
        batch_size=4,
        max_frame_size=1920,
        processes_count=2,
        use_og_extractor=False,
        face_extract_process_count=2,
        display_stream=True,
        seamless_clone=False,
    )

    video_path = "C:/Users/teckt/Documents/test_vids/iris2.mp4"
    swap_file(swapper,
              checkpoint_path="autoencoder_320_512.pth",
              video_path=video_path,
              output_path=f"{video_path}_swapped_gehlee.mp4",
              secs_to_stop_at=600)

    # swap_dir(swapper,
    #          checkpoint_path="autoencoder_320_512.pth",
    #          dir_path="C:/Users/teckt/Documents/webcam/sae",
    #          out_dir="C:/Users/teckt/Documents/test_vids/bo_swapped",
    #          replace_existing=False,
    #          secs_to_stop_at=5)

