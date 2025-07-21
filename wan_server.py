import datetime
import os
import sys
import time
import cv2
from PIL import Image

from wan import VideoAnnotator, WanVideoGenerator, WanSettings
from diffusers.utils.export_utils import _legacy_export_to_video
from redresser_utils import RedresserSettings
from fire_functions import FirestoreFunctions

from CONSTANTS import *


class RepaintJobProcesser:

    @staticmethod
    def make_dirs(job_dir=os.path.join('C:' + os.sep, 'deepfakes', 'df-maker-files')):
        RepaintJobProcesser.JOB_DIR = job_dir
        if not os.path.exists(RepaintJobProcesser.JOB_DIR):
            os.mkdir(RepaintJobProcesser.JOB_DIR)

        RepaintJobProcesser.USERS_DIR = os.path.join(RepaintJobProcesser.JOB_DIR, "users")
        if not os.path.exists(RepaintJobProcesser.USERS_DIR):
            os.mkdir(RepaintJobProcesser.USERS_DIR)

        RepaintJobProcesser.PUBLIC_MODELS_DIR = os.path.join(RepaintJobProcesser.JOB_DIR, "models", "dfs")
        RepaintJobProcesser.PRETRAINED_MODELS_DIR = os.path.join(RepaintJobProcesser.PUBLIC_MODELS_DIR, "pretrained")
        RepaintJobProcesser.BASE_MODELS_DIR = os.path.join(RepaintJobProcesser.PUBLIC_MODELS_DIR, "base")

    CHECKPOINT_FILE_NAME = "dfsModel.ckpt"


def setup_wan():
    wan = WanVideoGenerator(is_server=False, )
    wan.settings = WanSettings()
    wan.settings.options = WanSettings.default_options.copy()
    wan.settings.options["image_mod_value"] = wan.pipe.vae_scale_factor_spatial * \
                                              wan.pipe.transformer.config.patch_size[1]  # latent dims

    image_proc = VideoAnnotator()

    return wan, image_proc


def update_progress(pipeline, i, t, callback_kwargs):
    # print("i", i, "t", t,)

    # increment steps
    WanVideoGenerator.STEPS_CURRENT = (i + 1) * (WanVideoGenerator.NUM_FRAMES_ITER / WanVideoGenerator.NUM_FRAMES_MAX)

    FirestoreFunctions.wanVideoJobsRef.document(WanVideoGenerator.JOB_ID).set(
        {
            "job_progress": int(100 * (WanVideoGenerator.STEPS_CURRENT / WanVideoGenerator.STEPS_MAX)),
        }, merge=True
    )

    return callback_kwargs


def on_submit(
        wan, image_proc, firebase_token, firebase_uid, input_type, prompt, num_frames, steps, flow_shift, height, width,
        runs,
        source_image=None, source_video=None, frame_image=None, slider_value=None,
        add_video_control=False, control_video=None, start_frame=0, frame_skip=0, control_type="Full Pose", ref=None,
        seed=-1, conditioning_scale=1.0,
):
    start_time = time.time()

    print("Input Type:", input_type)
    print("Prompt:", prompt)
    print("Num Frames:", num_frames)
    print("Steps:", steps)
    print("Flow Shift:", flow_shift)
    print("Height:", height)
    print("Width:", width)
    print("Reference Images:", ref)
    print("Source Image:", source_image)
    print("Source Video:", source_video)
    print("Slider Value:", slider_value)
    print("Add Video Control:", add_video_control)
    print("Control Video:", control_video)
    print("Start Frame:", start_frame)
    print("Frame Skip:", frame_skip)
    print("Control Type:", control_type)
    print("Runs:", runs)
    print("Seed:", seed)
    print("Conditioning Scale:", conditioning_scale)

    wan.settings.options["width"] = width
    wan.settings.options["height"] = height
    wan.settings.options["max_area"] = height * width
    wan.settings.options["guidance_scale"] = 1.0
    wan.settings.options["num_inference_steps"] = steps
    wan.settings.options["flow_shift"] = flow_shift
    wan.settings.options["prompt"] = prompt

    wan.settings.options["conditioning_scale"] = conditioning_scale
    wan.settings.options["seed"] = seed

    wan.settings.options["callback_on_step_end"] = update_progress

    # wan needs 16 + 1 frames minimum, so we add 1 to num_frames; also we discard the first frame; chunks are always 81 but we discard the first one; users don't need to know this so igts just 80
    total_num_frames = num_frames + 1

    chunk_size = 32
    if add_video_control and control_video is not None:
        # we can chunk the video into 32 + 1 frames if we have a control video, so we set num_frames to 32 + 1
        wan.settings.options["num_frames"] = chunk_size + 1
    else:
        wan.settings.options["num_frames"] = total_num_frames

    if ref:
        reference_images = [Image.open(img) for img in ref if img is not None]
    else:
        reference_images = None

    WanVideoGenerator.NUM_FRAMES_MAX = total_num_frames - 1
    WanVideoGenerator.NUM_FRAMES_ITER = wan.settings.options["num_frames"] - 1
    WanVideoGenerator.STEPS_MAX = steps
    WanVideoGenerator.STEPS_CURRENT = 0

    FirestoreFunctions.wanVideoJobsRef.document(WanVideoGenerator.JOB_ID).set(
        {
            "job_progress_info": "Applying settings...",
        }, merge=True
    )

    # set fps and segment id/models for control video
    if add_video_control and control_video is not None:
        cap = cv2.VideoCapture(control_video)
        video_fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        cap.release()

        output_fps = int(video_fps / (frame_skip + 1))

        print(f"video_fps: {video_fps}, output_fps: {output_fps}")

        # map seg id
        '''
        choices=["Full Pose", "Person Mask", "Clothing Mask", "Face Mask", "Background Mask", "Depth Mask"]

        SEGMENT_FASHION = 0
        SEGMENT_PERSON = 1
        SEGMENT_FACE = 2  # Ignores keep_face; FaceGen
        SEGMENT_ALL = 3  # masks whole image (just white image) unless keep_face or keep_hands is True
        POSE_FULL = 4  # replaces whole image; no mask
        SEGMENT_BG = 5  # inverse person
        SEGMENT_DEPTH = 6  # replaces whole image; no mask
        '''
        if control_type == "Full Pose":
            wan.settings.options["SEGMENT_ID"] = RedresserSettings.POSE_FULL
        elif control_type == "Person Mask":
            wan.settings.options["SEGMENT_ID"] = RedresserSettings.SEGMENT_PERSON
        elif control_type == "Clothing Mask":
            wan.settings.options["SEGMENT_ID"] = RedresserSettings.SEGMENT_FASHION
        elif control_type == "Face Mask":
            wan.settings.options["SEGMENT_ID"] = RedresserSettings.SEGMENT_FACE
        elif control_type == "Background Mask":
            wan.settings.options["SEGMENT_ID"] = RedresserSettings.SEGMENT_BG
        elif control_type == "Depth Mask":
            wan.settings.options["SEGMENT_ID"] = RedresserSettings.SEGMENT_DEPTH
        # elif control_type == "All Mask":  # well, this doesn't make sense because you might as well do T2V
        #     wan.settings.options["SEGMENT_ID"] = RedresserSettings.SEGMENT_ALL
        wan.settings.options["keep_face"] = False
        wan.settings.options["keep_hands"] = False

        if wan.settings.options["SEGMENT_ID"] != RedresserSettings.POSE_FULL:
            image_proc.set_seg_models(wan.settings)

    # predict loop

    for i in range(runs):
        all_video_frames = []
        iterations = 0
        num_frames_processed = 0
        reached_end_of_video = False
        # stash all source video frames before the starting frame (adds all frames if slider_value(frame_index) is total_frames-1)
        if input_type == "LF2V" and source_video is not None:
            cap = cv2.VideoCapture(source_video)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if slider_value is None or slider_value < 0:
                slider_value = 0
            elif slider_value >= total_frames:
                slider_value = total_frames - 1

            print(f"Total frames in control video: {total_frames}, ending at frame {slider_value}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # we want to add all frames up to and including slider_value (we minus 1 because we incremented it from reading a frame)
                if cap.get(cv2.CAP_PROP_POS_FRAMES) - 1 <= slider_value:
                    # convert to rgb and 1.0
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame / 255
                    all_video_frames.append(frame)
                else:
                    break
            cap.release()
            print(f"Stashed {len(all_video_frames)} frames from source video up to frame {slider_value}")

        while True:

            # if no control video, do not chunk (the video won't do anything as there aren't enough frames if chunking at 16)
            # prepare control frames and mask
            if add_video_control and control_video is not None:
                FirestoreFunctions.wanVideoJobsRef.document(WanVideoGenerator.JOB_ID).set(
                    {
                        "job_progress_info": "Preparing controls...",
                    }, merge=True
                )
                if num_frames == 0:
                    # run until control video ends
                    wan.settings.options[
                        "num_frames"] = chunk_size + 1  # process whole control video in 80 frame chunks
                else:
                    wan.settings.options["num_frames"] = total_num_frames - chunk_size * iterations

                video, mask, reached_end_of_video = image_proc.prepare_video(
                    control_video,
                    wan.settings, start_frame=start_frame, frames=wan.settings.options["num_frames"],
                    frame_skip=frame_skip,
                    segment_id=wan.settings.options["SEGMENT_ID"]
                )
                if len(video) < 5:
                    print(
                        f"reached end of video, cannot process unless frames are in multiples of 4 + 1, remainder frames={len(video)}")
                    break

                # make sure video is in multiples of 4 + 1 and adjust num_frames accordingly
                if len(video) < wan.settings.options['num_frames']:
                    num_frames_mod = ((len(video) - 1) % 4)  # mod after subtracting 1
                    if num_frames_mod > 0:
                        video = video[:-num_frames_mod]
                        mask = mask[:-num_frames_mod]
                    wan.settings.options["num_frames"] = len(video)
                    print(f"adjusted num_frames [{wan.settings.options['num_frames']}] to {len(video)}")

                # replace first frame with previous frame or slider_value(frame_index) of source video
                if len(all_video_frames) > 0:

                    prev_frame = all_video_frames[-1].copy() * 255
                    video[0] = Image.fromarray(prev_frame.astype("uint8"))
                    if wan.settings.options["SEGMENT_ID"] != RedresserSettings.POSE_FULL:
                        mask[0] = Image.new("RGB", (video[0].width, video[0].height), 0)
            FirestoreFunctions.wanVideoJobsRef.document(WanVideoGenerator.JOB_ID).set(
                {
                    "job_progress_info": "Processing...",
                }, merge=True
            )
            if input_type == "T2V":
                if add_video_control and control_video is not None:
                    output_frames, output_file_name = wan.run(video, mask, references=reference_images,
                                                              video_fps=output_fps)
                else:
                    output_frames, output_file_name = wan.run(references=reference_images, video_fps=16)

            elif input_type == "I2V":
                # for I2V, we need to prepare the image only for first iteration
                if iterations == 0:
                    base_frame = wan.load_and_prepare_image(source_image)  # takes numpy array or file path
                    frame_inserts = {}
                    frame_inserts[0] = base_frame
                    if add_video_control and control_video is not None:
                        # prepare video and mask with the frame inserts
                        _video, _mask = WanVideoGenerator.prepare_video_and_mask(
                            frame_inserts=frame_inserts,
                            height=base_frame.height, width=base_frame.width,
                            num_frames=wan.settings.options["num_frames"]
                        )
                        # replace control frames with frame inserts and masks
                        for k, v in frame_inserts.items():
                            video[k] = v
                            mask[k] = _mask[k]
                    else:
                        video, mask = WanVideoGenerator.prepare_video_and_mask(
                            frame_inserts=frame_inserts,
                            height=base_frame.height, width=base_frame.width,
                            num_frames=wan.settings.options["num_frames"]
                        )

                # if we're doing beyond the first iteration, it must have a control video and video+mask should've been processed
                output_frames, output_file_name = wan.run(video, mask, references=reference_images,
                                                          video_fps=16 if not add_video_control else output_fps)

            elif input_type == "LF2V":
                # for LF2V, we need to prepare the frame only for first iteration
                if iterations == 0:
                    base_frame = wan.load_and_prepare_image(frame_image)  # takes numpy array or file path

                    frame_inserts = {}
                    frame_inserts[0] = base_frame
                    video, mask = WanVideoGenerator.prepare_video_and_mask(
                        frame_inserts=frame_inserts,
                        height=base_frame.height, width=base_frame.width, num_frames=wan.settings.options["num_frames"]
                    )
                # if we're doing beyond the first iteration, it must have a control video
                output_frames, output_file_name = wan.run(video, mask, references=reference_images,
                                                          video_fps=16 if not add_video_control else output_fps)
                # export full video
            FirestoreFunctions.wanVideoJobsRef.document(WanVideoGenerator.JOB_ID).set(
                {
                    "job_progress_info": "Post processing...",
                }, merge=True
            )
            # post-process output frames
            # encode video for current iteration's output
            # _legacy_export_to_video(output_frames, output_file_name, fps=16 if not add_video_control else output_fps)

            # export full video by adding all output frames to all_video_frames and always discard first frame
            if len(all_video_frames) > 0:
                output_frames = output_frames[1:]

            for _f in output_frames:
                num_frames_processed += 1
                all_video_frames.append(_f)

            print(f"All video frames length: {len(all_video_frames)}, Output frames length: {len(output_frames)}")

            _legacy_export_to_video(all_video_frames, output_file_name + "_wan_full.mp4",
                                    fps=16 if not add_video_control else output_fps)
            # run_ffmpeg_optical_flow(input_video=video_path+"_wan_full.mp4", output_video=video_path+"_wan_full_opt.mp4", fps=int(video_fps/(frame_skip+1) * 2))

            if reached_end_of_video:
                print(f"reached end of video, remainder frames={len(video)}")
                break

            # check if we reached num_frames limit (0 means no limit)
            if num_frames > 0 and num_frames_processed >= total_num_frames:
                print(f"processed {num_frames_processed} frames, reached num_frames limit of {total_num_frames}")
                break

            if not add_video_control:
                # if no control video, we must break after one iteration (video will become incoherent)
                break

            # set the next start frame ( minus one because we're going to use the last frame as the first frame)
            if frame_skip > 0:
                start_frame = start_frame + len(video) * (frame_skip + 1) - (
                        frame_skip + 1)  # add one to frame skip (skipping 3 means every 4th)
            else:
                start_frame = start_frame + len(video) - 1

            iterations += 1

    return (output_file_name, output_file_name + "_wan_full.mp4")


def run(machine_id="OVERLORD4-0"):
    from pathlib import Path
    RepaintJobProcesser.make_dirs(os.path.join(Path().absolute(), JOB_DIR))
    print("JOB_DIR=", os.path.join(Path().absolute(), JOB_DIR))
    firestoreFunctions = FirestoreFunctions()
    FirestoreFunctions.machine_id = machine_id
    firestoreFunctions.machine_id = machine_id

    wan = image_proc = None

    job_order = 0
    job_orders = {
        0: FirestoreFunctions.WAN_VIDEO_JOB,
        # 0: FirestoreFunctions.TRAINING_JOB,
        1: FirestoreFunctions.WAN_VIDEO_JOB,
    }

    while True:

        total_secs = time.time()
        secs = time.time()

        job_order = job_order + 1 if job_order < len(job_orders) - 1 else 0
        job_type = job_orders[job_order]
        repaint_mode = job_order  # just happens to be the same

        # job_type = FirestoreFunctions.TRAINING_JOB if training_job_first else FirestoreFunctions.SWAPPING_JOB
        # job_type = FirestoreFunctions.REPAINT_IMAGE_JOB

        started_job = firestoreFunctions.get_started_jobs(job_type=job_type)
        if started_job is None:
            # print(f'getting locked jobs ({job_type}) to start...')
            locked_job = firestoreFunctions.get_locked_jobs(job_type=job_type)
            if locked_job is not None:
                print(f'starting locked job({locked_job.id}) ({job_type})...')
                firestoreFunctions.start_job(job_type=job_type, job=locked_job)
                print(f'job started! keeping same job type ({job_type}) for next loop, sleeping 1 second...')
                job_order -= 1
                sys.stdout.flush()
                time.sleep(0.25)
                continue
            else:
                job_to_lock = firestoreFunctions.get_jobs(job_type=job_type, resolutions=[], )
                if job_to_lock is not None:
                    print(f'\nlocking job({job_to_lock.id}) ({job_type})...')
                    firestoreFunctions.lock_job(job_type=job_type, job=job_to_lock)
                    print(f'job locked! keeping same job type ({job_type}) for next loop, sleeping 1 second...')
                    job_order -= 1
                    time.sleep(0.25)
                    firestoreFunctions.db.collection("wanServers").document(machine_id).set(
                        {
                            "lastActiveTime": datetime.datetime.now(datetime.timezone.utc),
                            "isBusy": True
                        }
                    )
                    continue
                else:
                    # Get the current date and time
                    current_time = datetime.datetime.now()

                    # Format and display the current time with AM/PM
                    formatted_time = current_time.strftime("%I:%M:%S %p")
                    print(f'\r[{formatted_time}]No jobs available for ({job_type}), sleeping 5 second...', end="")

                    firestoreFunctions.db.collection("wanServers").document(machine_id).set(
                        {
                            "lastActiveTime": datetime.datetime.now(datetime.timezone.utc),
                            "isBusy": False
                        }
                    )

                    time.sleep(5)
                    continue
        FirestoreFunctions.job_id = started_job.id

        # set the downloaded image path
        job_dict = started_job.to_dict()
        job_dict["id"] = started_job.id

        WanVideoGenerator.JOB_ID = started_job.id

        firestoreFunctions.db.collection("wanServers").document(machine_id).set(
            {
                "lastActiveTime": datetime.datetime.now(datetime.timezone.utc),
                "isBusy": True
            }
        )

        wan_args = retrieve_from_firebase(job_dict)
        # insert wan and image_proc in args beginning
        wan_args = list(wan_args)  # convert tuple to list

        if wan is None or image_proc is None:
            FirestoreFunctions.wanVideoJobsRef.document(WanVideoGenerator.JOB_ID).set(
                {
                    "job_progress_info": "Preparing models...",
                }, merge=True
            )
            wan, image_proc = setup_wan()

        wan_args.insert(0, wan)
        wan_args.insert(1, image_proc)

        result = on_submit(*wan_args)

        if not result:
            print("job failed?")
        else:
            # upload video result to storage and set job status to completed
            complete_job(firebase_uid=job_dict['userId'], job_id=job_dict["id"], result=result)
            print("job completed")

        job_order -= 1


def complete_job(firebase_uid, job_id, result):
    output_file_name, full_video_path = result
    user_path = f"users/{firebase_uid}/wanVideoJobs/{job_id}"

    # Upload output video
    FirestoreFunctions.send_job_file(user_path, full_video_path, "result_video.mp4", required=True)

    # Update Firestore document with job status
    while True:
        try:
            FirestoreFunctions.wanVideoJobsRef.document(job_id).set({
                'jobStatus': "ended",
                'endedTime': int(time.time()),
            }, merge=True)
            break
        except Exception as e:
            print(e, "Failed to update swapping doc jobStatus. retrying in 5 seconds...")
            time.sleep(5)


def retrieve_from_firebase(job_doc_data):
    # stores the job data from the Firestore document in same format as on_submit neccessary to run wan
    '''
    doc_data = {
        'user_id': user_id,
        'prompt': prompt,
        'input_type': input_type,
        'num_frames': num_frames,
        'steps': steps,
        'flow_shift': flow_shift,
        'height': height,
        'width': width,
        'runs': runs,
        'seed': seed,
        'conditioning_scale': conditioning_scale,
        'add_video_control': add_video_control,
        'control_type': control_type,
        'start_frame': start_frame,
        'frame_skip': frame_skip,
        'slider_value': slider_value,
        'storage_paths': storage_paths,
        'jobStatus': 'queued',
        'queuedTime': time.time()
    }
    '''
    job_id = job_doc_data.get('id', '')

    user_id = job_doc_data.get('userId', '')

    input_type = job_doc_data.get('input_type', 'T2V')
    prompt = job_doc_data.get('prompt', '')
    num_frames = job_doc_data.get('num_frames', 80)
    steps = job_doc_data.get('steps', 4)
    flow_shift = job_doc_data.get('flow_shift', 2.0)
    height = job_doc_data.get('height', 480)
    width = job_doc_data.get('width', 832)
    runs = job_doc_data.get('runs', 1)
    seed = job_doc_data.get('seed', -1)
    conditioning_scale = job_doc_data.get('conditioning_scale', 1.0)

    storage_paths = job_doc_data.get('storage_paths', {})
    print(f"storage_paths; {storage_paths}")

    ref = storage_paths.get('reference_images', None)
    print(f"ref; {ref}")

    source_image = storage_paths.get('source_image', None)
    source_video = storage_paths.get('source_video', None)
    frame_image = storage_paths.get('frame_image', None)
    slider_value = job_doc_data.get('slider_value', None)

    add_video_control = job_doc_data.get('add_video_control', False)
    control_video = storage_paths.get('control_video', None)
    control_type = job_doc_data.get('control_type', 'Full Pose')
    start_frame = job_doc_data.get('start_frame', 0)
    frame_skip = job_doc_data.get('frame_skip', 0)

    storage_root = f"users/{user_id}/wanVideoJobs/{job_id}"

    # download any images or videos from storage paths
    if source_image:
        download_destination_path = f"{storage_root}/source_image.png"
        # make the dir if not exists
        os.makedirs(os.path.dirname(download_destination_path), exist_ok=True)
        if FirestoreFunctions.exists_in_storage(source_image):
            FirestoreFunctions.chunked_download(source_image, download_destination_path)
        else:
            raise FileNotFoundError(f"Source image not found in storage: {source_image}")

    if source_video:
        download_destination_path = f"{storage_root}/source_video.mp4"
        os.makedirs(os.path.dirname(download_destination_path), exist_ok=True)
        if FirestoreFunctions.exists_in_storage(source_video):
            FirestoreFunctions.chunked_download(source_video, download_destination_path)
        else:
            raise FileNotFoundError(f"Source video not found in storage: {source_video}")

    if frame_image:
        download_destination_path = f"{storage_root}/frame_image.png"
        os.makedirs(os.path.dirname(download_destination_path), exist_ok=True)
        if FirestoreFunctions.exists_in_storage(frame_image):
            FirestoreFunctions.chunked_download(frame_image, download_destination_path)
        else:
            raise FileNotFoundError(f"Frame image not found in storage: {frame_image}")

    if ref:
        ref_images = []
        for idx, img in enumerate(ref):
            download_destination_path = f"{storage_root}/ref_{idx}.png"
            os.makedirs(os.path.dirname(download_destination_path), exist_ok=True)
            if FirestoreFunctions.exists_in_storage(img):
                FirestoreFunctions.chunked_download(img, download_destination_path)
                ref_images.append(download_destination_path)
            else:
                raise FileNotFoundError(f"Reference image {idx} not found in storage: {img}")

    if add_video_control and control_video:
        download_destination_path = f"{storage_root}/control_video.mp4"
        os.makedirs(os.path.dirname(download_destination_path), exist_ok=True)
        if FirestoreFunctions.exists_in_storage(control_video):
            FirestoreFunctions.chunked_download(control_video, download_destination_path)
        else:
            raise FileNotFoundError(f"Control video not found in storage: {control_video}")

    # now we return all args in the same format as on_submit (we don't need firebase token here)
    return (None, user_id, input_type, prompt, num_frames, steps, flow_shift, height, width, runs,
            source_image, source_video, frame_image, slider_value,
            add_video_control, control_video, start_frame, frame_skip, control_type, ref, seed, conditioning_scale)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--machine_id", default='OVERLORD4-0')

    args = parser.parse_args()

    machine_id = args.machine_id

    print(f"running with machine_id={machine_id}")

    # run("cog-i2v", is_server=False)
    run(machine_id)
