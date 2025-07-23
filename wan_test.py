'''
results come back as one video file
audio can be copied from a source video at a specific frame
no chunk editing but can continue/start on any frame (discards last frame)

pricing is per frame per size per steps
anything over 81 frames will be chunked into 80 frames per chunk (chunk_size -1 because we always discard first frame) and rounded down to the nearest multiple of four + 1 (model needs batches of 4 + 1)
formula is total_chunks = (frames-1) // (chunk_size-1)
actual frames = total_chunks * (chunk_size - 1) + 1

when selecting a video, the
'''

from diffusers.utils.export_utils import _legacy_export_to_video
from redresser_utils import RedresserSettings
from wan import VideoAnnotator, WanVideoGenerator, WanSettings
from fire_functions import FirestoreFunctions
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import time
import uuid
import os
import random
from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

MAX_ACTIVE_JOBS = 4


def get_frame(video_path, frame_idx):
    print(f"Getting frame {frame_idx} from {video_path}")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def get_total_frames(video_path):
    print(f"Getting total frames from {video_path}")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def update_slider(video_path):
    if video_path is None:
        return gr.update(maximum=1, value=0), gr.update(value=None)
    total = get_total_frames(video_path)
    return gr.update(maximum=total - 1, value=total - 1), gr.update(value=get_frame(video_path, total - 1))


def update_frame(video_path, frame_idx):
    if video_path is None:
        return None
    return get_frame(video_path, frame_idx)


def make_source_video_slider(visible=True):

    source_video = gr.Video(label="Source Video", visible=visible)

    frame_image = gr.Image(label="Source Frame", type="filepath", visible=visible)
    slider = gr.Slider(
        minimum=0,
        maximum=1,
        step=1,
        value=0,
        label="Frame Index",
        interactive=True,
        show_label=True,
        elem_id="source_frame_slider",
        visible=visible
    )

    # When video is loaded, update/show slider max and frame image to last frame of video and hide the video
    source_video.change(
        update_slider,
        inputs=source_video,
        outputs=[slider, frame_image]
    )
    # When slider changes, update frame image
    slider.change(
        update_frame,
        inputs=[source_video, slider],
        outputs=frame_image
    )

    return source_video, frame_image, slider


def make_universal_options(input_type):
    # universal WAN video options

    gr.Markdown("### Universal Options")
    prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here", lines=4,)
    if input_type == "LF2V":
        num_frames = gr.Number(label="Num Frames (0 to process whole control video in 80 frame chunks)", value=81,
                               precision=0, interactive=True)
    else:
        num_frames = gr.Number(label="Num Frames", value=81, precision=0, interactive=True)

    with gr.Row():
        gr.Markdown("### Reference Images (up to 4)")
        add_ref_btn = gr.Button("Add Reference Image")

    ref_images = gr.State([])  # Store the list of reference images
    ref_image_boxes = []
    with gr.Row() as ref_images_row:
        for i in range(4):
            img = gr.Image(label=f"Reference Image {i + 1}", visible=False, interactive=True, type="filepath")
            ref_image_boxes.append(img)

    def add_ref_image(refs):
        # Find the first invisible image box and make it visible
        for idx, img in enumerate(refs):
            if not img:
                refs[idx] = True
                break
        return *[gr.update(visible=v) for v in refs], refs

    # Track which image boxes are visible (True/False)
    ref_visible = gr.State([False, False, False, False])

    add_ref_btn.click(
        add_ref_image,
        inputs=ref_visible,
        outputs=[*ref_image_boxes, ref_visible]
    )


def make_control_options():
    # Control video options

    gr.Markdown("### Control Options")

    add_video_control = gr.Checkbox(label="Add Control Video", value=False)
    with gr.Row(visible=False) as video_control_options:
        with gr.Column():
            video = gr.Video(label="Control Video")
        with gr.Column():
            start_frame = gr.Number(label="Start Control Frame", value=0, precision=0, interactive=True)
            frame_skip = gr.Number(label="Frame Skip", value=0, precision=0, interactive=True)
            control_type = gr.Dropdown(
                label="Control Type",
                choices=["Full Pose", "Person Mask", "Clothing Mask", "Face Mask", "Background Mask"],
                value="Full Pose",
                interactive=True
            )

    add_video_control.change(
        lambda v: gr.update(visible=v),
        inputs=add_video_control,
        outputs=video_control_options
    )

    # def make_tab(input_type, wan):
    with gr.Blocks() as block:
        with gr.Row():
            with gr.Column():
                make_universal_options()
                if input_type == "I2V":
                    gr.Markdown("### Image Source Options (Select an image to generate from)")
                    source_image = gr.Image(label="Source Image", type="filepath")
                elif input_type == "LF2V":
                    gr.Markdown(
                        "### Video Source Options (Select a video to select a specific frame to generate/continue from)")
                    source_video, slider = make_source_video_slider()
            with gr.Column():
                make_control_options()
    return block

    # Video from text (T2V), video source (generates from last frame; LF2V), image source (I2V)
    # 3 tabs: T2V, I2V, LF2V
    # T2V should be prompt + reference images
    # I2V should be prompt + reference images + source image
    # LF2V should be prompt + reference images + video source (to generate from last frame)
    # Differences are that T2V only has reference images, while I2V and LF2V has a source image and source video respectively.

    # The rest of the control options are the same for all three (T+R | T+R+I | T+R+V) + CV + CF
    # Control Video can select from "Full Pose", "Person Mask", "Clothing Mask", "Face Mask", "Background Mask"
    # Control Frames are images to transition to at the designated frame index (0 to num_frames) and takes precedence over anything else (0 would just be I2V)

    # after generating the video, show the result and have options to copy to LF2V (replaces LF2V source video)


def calculate_output_size(height, width, object, latent_dims=16):
    if isinstance(object, str):
        # If object is a string, it is a path to a video
        cap = cv2.VideoCapture(object)
        if not cap.isOpened():
            return "Output Size: -"
        object_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        object_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    elif isinstance(object, np.ndarray):
        # If object is an image, get its dimensions (numpy array)
        object_height, object_width = object.shape[:2]
    else:
        # If object is not a video or image, return default
        return "Output Size: -"

    # Use defaults if not set
    if not height or not width:
        return "Output Size: -"

    max_area = height * width
    aspect_ratio = object_height / object_width
    height_calc = round(np.sqrt(max_area * aspect_ratio)) // latent_dims * latent_dims
    width_calc = round(np.sqrt(max_area / aspect_ratio)) // latent_dims * latent_dims
    return f"Output Size: {height_calc} x {width_calc}"


def make_tab(input_type):
    with gr.Blocks() as block:
        with gr.Row():
            with gr.Column():
                # Universal options
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here", lines=4)
                with gr.Row():
                    if input_type == "LF2V":
                        num_frames = gr.Number(label="Num Frames (0 to process whole control video in 80 frame chunks)",
                                            value=80, precision=0, interactive=True, minimum=0, step=16)
                    else:
                        num_frames = gr.Number(label="Num Frames",
                                            value=80, precision=0, interactive=True, minimum=0, step=16)

                    steps = gr.Number(label="Steps", value=4, precision=0, interactive=True, minimum=4, maximum=8)
                flow_shift = gr.Number(label="Flow Shift", value=2.0, precision=1, interactive=True, minimum=1.0,
                                    maximum=5.0, step=0.1)

                if input_type == "I2V":
                    gr.Markdown("### Output will be scaled to HxW using aspect ratio of source image")
                if input_type == "LF2V":
                    gr.Markdown("### Output will be scaled to HxW using aspect ratio of source video")

                # put height and width here if t2v only; otherwise will be under source image/video
                if input_type == "T2V":
                    with gr.Row():
                        height = gr.Number(label="Height", value=480, precision=0, interactive=True, minimum=240,
                                        maximum=896, step=16)
                        width = gr.Number(label="Width", value=832, precision=0, interactive=True, minimum=240, maximum=896,
                                        step=16)

                conditioning_scale_input = gr.Number(label="Strength (For reference images, I2V, and LF2V and Control Video)", value=1.0, precision=1, interactive=True,
                                               minimum=0.0, maximum=1.0, step=0.1)
                with gr.Row():
                    seed_checkbox = gr.Checkbox(label="Use Random Seed", value=True, interactive=True)
                    seed_input = gr.Number(label="Seed", value=-1, precision=0, interactive=True, minimum=-1, maximum=4294967294, visible=False)

                seed_checkbox.change(
                    lambda v: gr.update(visible=True, value=int(random.randrange(4294967294))) if not v else gr.update(visible=False, value=-1),
                    inputs=seed_checkbox,
                    outputs=[seed_input]
                )

                # Reference images
                gr.Markdown("### Reference Images (up to 4)")
                add_ref_btn = gr.Button("Add Reference Image")
                ref_image_boxes = []
                with gr.Row() as ref_images_row:
                    for i in range(4):
                        img = gr.Image(label=f"Reference Image {i + 1}", visible=False, interactive=True, type="filepath")
                        ref_image_boxes.append(img)
                ref_visible = gr.State([False, False, False, False])

                def add_ref_image(refs):
                    for idx, img in enumerate(refs):
                        if not img:
                            refs[idx] = True
                            break
                    return *[gr.update(visible=v) for v in refs], refs

                add_ref_btn.click(
                    add_ref_image,
                    inputs=ref_visible,
                    outputs=[*ref_image_boxes, ref_visible]
                )



                # Source image or video

                if input_type == "I2V":
                    gr.Markdown("### Image Source Options (Select an image to generate from)")
                    source_image = gr.Image(label="Source Image", type="filepath")
                else:
                    source_image = gr.Image(label="Source Image", type="filepath", visible=False)

                if input_type == "LF2V":
                    gr.Markdown(
                        "### Video Source Options (Select a video, then select a frame to generate/continue from)")
                    source_video, frame_image, slider = make_source_video_slider(visible=True)
                else:
                    source_video, frame_image, slider = make_source_video_slider(visible=False)

                # Height and Width for I2V and LF2V
                if input_type in ["I2V", "LF2V"]:
                    with gr.Row():
                        height = gr.Number(label="Height", value=480, precision=0, interactive=True, minimum=240,
                                        maximum=920, step=16)
                        width = gr.Number(label="Width", value=832, precision=0, interactive=True, minimum=240, maximum=920,
                                        step=16)
                    output_size_display = gr.Markdown("Output Size: -")
                    height.change(
                        calculate_output_size,
                        inputs=[height, width,
                                source_video if input_type == "LF2V" else source_image],
                        outputs=output_size_display
                    )
                    width.change(
                        calculate_output_size,
                        inputs=[height, width,
                                source_video if input_type == "LF2V" else source_image],
                        outputs=output_size_display
                    )

                    if input_type == "I2V":
                        source_image.change(
                            calculate_output_size,
                            inputs=[height, width,
                                    source_video if input_type == "LF2V" else source_image],
                            outputs=output_size_display
                        )
                    else:
                        source_video.change(
                            calculate_output_size,
                            inputs=[height, width,
                                    source_video if input_type == "LF2V" else source_image],
                            outputs=output_size_display
                        )

            with gr.Column():
                # Control options
                gr.Markdown("### Control Options")
                add_video_control = gr.Checkbox(label="Add Control Video", value=False)
                with gr.Row(visible=False) as video_control_options:
                    with gr.Column():

                        control_video = gr.Video(label="Control Video")
                        control_type = gr.Dropdown(
                            label="Control Type",
                            choices=["Full Pose", "Person Mask", "Clothing Mask", "Face Mask", "Background Mask",
                                     "Depth Mask"],
                            value="Full Pose",
                            interactive=True
                        )
                        with gr.Row():
                            start_frame = gr.Number(label="Start Control Frame", value=0, precision=0, interactive=True,
                                                    minimum=0)
                            frame_skip = gr.Number(label="Frame Skip", value=0, precision=0, interactive=True, minimum=0,
                                                maximum=3)

                add_video_control.change(
                    lambda v: gr.update(visible=v),
                    inputs=add_video_control,
                    outputs=video_control_options
                )

        # --- Submit Button ---
        runs = gr.Number(label="Runs", value=1, precision=0, interactive=True, minimum=1, maximum=1, visible=False)
        submit_btn = gr.Button("Submit", variant="primary")

        # result video output

        result_info   = gr.Markdown("**⏳ Your job is processing…**", visible=False)
        history_button = gr.Button("Go to Jobs", visible=True)

        # Prepare input list for submit button
        submit_inputs = [prompt, num_frames, steps, flow_shift, height, width, runs]
        submit_inputs.extend([source_image, source_video, frame_image, slider])
        submit_inputs.extend([
            add_video_control, control_video, start_frame, frame_skip, control_type, *ref_image_boxes
        ])
        submit_inputs.extend([
            conditioning_scale_input, seed_input
        ])

        # if input_type == "I2V":
        #     submit_inputs.extend([source_image])
        #     submit_inputs.extend([
        #         add_video_control, control_video, start_frame, frame_skip, control_type, *ref_image_boxes
        #     ])

        # elif input_type == "LF2V":
        #     submit_inputs.extend([source_video, frame_image, slider])
        #     submit_inputs.extend([
        #         add_video_control, control_video, start_frame, frame_skip, control_type, *ref_image_boxes
        #     ])

        # else:
        #     submit_inputs.extend([
        #         add_video_control, control_video, start_frame, frame_skip, control_type, *ref_image_boxes
        #     ])

    return block, submit_inputs, submit_btn, source_image, source_video, result_info, history_button


def on_submit_t2v(
        firebase_token, firebase_uid, prompt, num_frames, steps, flow_shift, height, width, runs,
        source_image=None, source_video=None, frame_image=None, slider_value=None,
        add_video_control=False, control_video=None, start_frame=0, frame_skip=0, control_type="Full Pose",
        ref_0=None, ref_1=None, ref_2=None, ref_3=None, conditioning_scale=1.0, seed=-1
):
    ref_image_boxes = [ref_0, ref_1, ref_2, ref_3]
    return on_submit(
        firebase_token, firebase_uid, "T2V", prompt, num_frames, steps, flow_shift, height, width, runs,
        source_image=None, source_video=None, frame_image=None, slider_value=None,
        add_video_control=add_video_control, control_video=control_video, start_frame=start_frame,
        frame_skip=frame_skip, control_type=control_type, ref=ref_image_boxes,
        conditioning_scale=conditioning_scale, seed=seed,
    )


def on_submit_i2v(
        firebase_token, firebase_uid, prompt, num_frames, steps, flow_shift, height, width, runs,
        source_image=None, source_video=None, frame_image=None, slider_value=None,
        add_video_control=False, control_video=None, start_frame=0, frame_skip=0, control_type="Full Pose",
        ref_0=None, ref_1=None, ref_2=None, ref_3=None, conditioning_scale=1.0, seed=-1
):
    ref_image_boxes = [ref_0, ref_1, ref_2, ref_3]
    return on_submit(
        firebase_token, firebase_uid, "I2V", prompt, num_frames, steps, flow_shift, height, width, runs,
        source_image=source_image, source_video=None, frame_image=None, slider_value=None,
        add_video_control=add_video_control, control_video=control_video, start_frame=start_frame,
        frame_skip=frame_skip, control_type=control_type, ref=ref_image_boxes,
        conditioning_scale=conditioning_scale, seed=seed
    )


def on_submit_lf2v(
        firebase_token, firebase_uid, prompt, num_frames, steps, flow_shift, height, width, runs,
        source_image=None, source_video=None, frame_image=None, slider_value=None,
        add_video_control=False, control_video=None, start_frame=0, frame_skip=0, control_type="Full Pose",
        ref_0=None, ref_1=None, ref_2=None, ref_3=None, conditioning_scale=1.0, seed=-1
):
    ref_image_boxes = [ref_0, ref_1, ref_2, ref_3]
    return on_submit(
        firebase_token, firebase_uid, "LF2V", prompt, num_frames, steps, flow_shift, height, width, runs,
        source_image=None, source_video=source_video, frame_image=frame_image, slider_value=slider_value,
        add_video_control=add_video_control, control_video=control_video, start_frame=start_frame,
        frame_skip=frame_skip, control_type=control_type, ref=ref_image_boxes,
        conditioning_scale=conditioning_scale, seed=seed
    )


def on_submit(
        firebase_token, firebase_uid, input_type, prompt, num_frames, steps, flow_shift, height, width, runs,
        source_image=None, source_video=None, frame_image=None, slider_value=None,
        add_video_control=False, control_video=None, start_frame=0, frame_skip=0, control_type="Full Pose", ref=None,
        conditioning_scale=1.0, seed=-1
):
    submit_btns = [gr.update(interactive=False) for _ in range(3)]
    result_infos = [gr.update(visible=True) for _ in range(3)]
    history_buttons = [gr.update(visible=True) for _ in range(3)]

    start_time = time.time()

    authenticated, response_message = FirestoreFunctions.authenticate_user(firebase_token, firebase_uid)
    # if not authenticated:
    #     return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=response_message)
    # else:
    #     submit_to_firebase()

    try:
        submit_to_firebase(firebase_uid, input_type, prompt, num_frames, steps, flow_shift, height, width, runs,
            source_image=source_image, source_video=source_video, frame_image=frame_image, slider_value=slider_value,
            add_video_control=add_video_control, control_video=control_video, start_frame=start_frame, frame_skip=frame_skip, control_type=control_type, ref=ref,
            conditioning_scale=conditioning_scale, seed=seed)

    except Exception as e:
        print(f"Error submitting job: {e}")
        submit_btns = [gr.update(interactive=True) for _ in range(3)]
        result_infos = [gr.update(visible=True, value=str(e)) for _ in range(3)]
        history_buttons = [gr.update(visible=False) for _ in range(3)]

    # we need to update submit buttons, result infos, and go to history buttons
    return *submit_btns, *result_infos, *history_buttons, gr.update(active=True)


def submit_to_firebase(
        firebase_uid, input_type, prompt, num_frames, steps, flow_shift, height, width, runs,
        source_image=None, source_video=None, frame_image=None, slider_value=None,
        add_video_control=False, control_video=None, start_frame=0, frame_skip=0, control_type="Full Pose", ref=None,
        conditioning_scale=1.0, seed=-1
    ):

    print("Submitting to Firebase...")
    print("Firebase UID:", firebase_uid)
    print("Input Type:", input_type)
    print("Prompt:", prompt)
    print("Num Frames:", num_frames)
    print("Steps:", steps)
    print("Flow Shift:", flow_shift)
    print("Height:", height)
    print("Width:", width)
    print("Runs:", runs)
    print("Reference Images:", ref)
    print("Source Image:", source_image)
    print("Source Video:", source_video)
    print("Slider Value:", slider_value)
    print("Frame Image:", frame_image)
    print("Add Video Control:", add_video_control)
    print("Control Video:", control_video)
    print("Start Frame:", start_frame)
    print("Frame Skip:", frame_skip)
    print("Control Type:", control_type)

    print("Conditioning Scale:", conditioning_scale)
    print("Seed:", seed)

    # Prepare file paths
    # prepare the job_id from uuid
    job_id = str(uuid.uuid4())

    user_path = f"users/{firebase_uid}/wanVideoJobs/{job_id}"
    storage_paths = {}

    # Upload source image
    if source_image:
        img_remote = f"{user_path}/source_image.png"
        FirestoreFunctions.send_job_file(user_path, source_image, "source_image.png", required=True)
        storage_paths['source_image'] = img_remote

    # Upload source video
    if source_video:
        vid_remote = f"{user_path}/source_video.mp4"
        FirestoreFunctions.send_job_file(user_path, source_video, "source_video.mp4", required=True)
        storage_paths['source_video'] = vid_remote

    # Upload frame image
    if frame_image:
        frame_remote = f"{user_path}/frame_image.png"
        FirestoreFunctions.send_job_file(user_path, frame_image, "frame_image.png", required=True)
        storage_paths['frame_image'] = frame_remote

    # Upload any reference images
    if ref:
        storage_paths['reference_images'] = []
        for idx, img in enumerate(ref):
            if img is None:
                continue

            ref_remote = f"{user_path}/ref_{idx}.png"
            FirestoreFunctions.send_job_file(user_path, img, f"ref_{idx}.png", required=True)
            storage_paths['reference_images'].append(ref_remote)

    # Upload control video if present
    if add_video_control and control_video:
        ctrl_remote = f"{user_path}/control_video.mp4"
        FirestoreFunctions.send_job_file(user_path, control_video, "control_video.mp4", required=True)
        storage_paths['control_video'] = ctrl_remote

    # Build Firestore document
    doc_ref = FirestoreFunctions.db.collection('wanVideoJobs').document(job_id)
    doc_data = {
        'userId': firebase_uid,
        'prompt': prompt,
        'input_type': input_type,
        'num_frames': num_frames,
        'steps': steps,
        'flow_shift': flow_shift,
        'height': height,
        'width': width,
        'runs': runs,
        'add_video_control': add_video_control,
        'control_type': control_type,
        'start_frame': start_frame,
        'frame_skip': frame_skip,
        'slider_value': slider_value,
        'storage_paths': storage_paths,
        'jobStatus': 'queued',
        'queuedTime': time.time(),
        'conditioning_scale': conditioning_scale,
        'seed': seed,
    }
    doc_ref.set(doc_data)
    print("Firestore document created with ID:", job_id)
    print("Document data:", doc_data)


PAGE_SIZE = 10


def fetch_jobs(firebase_uid, firebase_token, page=0, raw=False):

    # Authenticate user
    # authenticated, response_message = FirestoreFunctions.authenticate_user(firebase_token, firebase_uid)

    query = FirestoreFunctions.db.collection("wanVideoJobs") \
              .where(filter=FieldFilter("userId", "==", firebase_uid))

    query = query.order_by("queuedTime", direction=firestore.firestore.Query.DESCENDING) \
                 .offset(int(page) * PAGE_SIZE).limit(PAGE_SIZE)

    docs = query.get()
    rows = []
    if raw:
        # for doc in docs:
        #     queuedTime = int(data.get("queuedTime", 0)) if data.get("queuedTime") else 0
        #     human_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(queuedTime))
        #     print(f"Job ID: {doc.id}, Queued Time: {human_time}")
        return docs

    for doc in docs:
        data = doc.to_dict()
        queuedTime = int(data.get("queuedTime", 0)) if data.get("queuedTime") else 0
        startedTime = int(data.get("startedTime", 0)) if data.get("startedTime") else 0
        endedTime = int(data.get("endedTime", 0)) if data.get("endedTime") else 0
        steps = int(data.get("steps", 0)) if data.get("steps") else 0
        height = int(data.get("height", 0)) if data.get("height") else 0
        width = int(data.get("width", 0)) if data.get("width") else 0
        rows.append([
            queuedTime,
            data.get("jobStatus", ""),
            data.get("input_type", ""),
            data.get("prompt", ""),
            data.get("num_frames", 0),
            steps,
            f"{height} x {width}",
            doc.id,
        ])
    print(f"Fetched {len(rows)} jobs for page {page}.")
    print("Rows:", rows)
    return rows


def update_page(firebase_uid, firebase_token, page, direction):
    new_page = max(0, page + direction)
    # jobs = fetch_jobs(firebase_uid, firebase_token, new_page)
    # return gr.update(value=jobs), gr.update(value=new_page)
    return gr.update(value=new_page)


def update_page_prev(firebase_uid, firebase_token, page):
    return update_page(firebase_uid, firebase_token, page, -1)


def update_page_next(firebase_uid, firebase_token, page):
    return update_page(firebase_uid, firebase_token, page, 1)


def extract_settings_from_job(job_id):
    job = FirestoreFunctions.db.collection('wanVideoJobs').document(job_id).get()

    job_data = job.to_dict()

    user_id = job.get("userId")

    input_type = job_data.get("input_type")
    prompt = job_data.get("prompt", "")
    num_frames = job_data.get("num_frames", 80)
    steps = job_data.get("steps", 4)
    flow_shift = job_data.get("flow_shift", 2.0)
    height = job_data.get("height", 480)
    width = job_data.get("width", 832)
    runs = job_data.get("runs", 1)
    source_image = job_data.get("storage_paths", {}).get("source_image")
    source_image = download_source_image(user_id, job_id) if source_image else None
    source_video = job_data.get("storage_paths", {}).get("source_video")
    source_video = download_source_video(user_id, job_id) if source_video else None
    frame_image = job_data.get("storage_paths", {}).get("frame_image")
    frame_image = download_frame_image(user_id, job_id) if frame_image else None
    slider_value = job_data.get("slider_value", 0)
    add_video_control = job_data.get("add_video_control", False)
    control_video = job_data.get("storage_paths", {}).get("control_video")
    control_video = download_control_video(user_id, job_id) if control_video else None
    start_frame = job_data.get("start_frame", 0)
    frame_skip = job_data.get("frame_skip", 0)
    control_type = job_data.get("control_type", "Full Pose")
    conditioning_scale = job_data.get("conditioning_scale", 1.0)
    seed = job_data.get("seed", -1)

    ref_images = [None] * 4
    job_ref_images = job_data.get("storage_paths", {}).get("reference_images", [])
    ref_images = [
        download_reference_image(user_id, job_id, idx) if idx < len(job_ref_images) else None
        for idx in range(4)
    ]

    settings = {
        "input_type": input_type,
        "prompt": prompt,
        "num_frames": num_frames,
        "steps": steps,
        "flow_shift": flow_shift,
        "height": height,
        "width": width,
        "runs": runs,
        "source_image": source_image,
        "source_video": source_video,
        "frame_image": frame_image,
        "slider_value": slider_value,
        "add_video_control": add_video_control,
        "control_video": control_video,
        "start_frame": start_frame,
        "frame_skip": frame_skip,
        "control_type": control_type,
        "ref_images": ref_images,
        "conditioning_scale": conditioning_scale,
        "seed": seed
    }
    print("Extracted settings from job:", settings)

    return (
        input_type,
        prompt,
        num_frames,
        steps,
        flow_shift,
        height,
        width,
        runs,
        source_image,
        source_video,
        frame_image,
        slider_value,
        add_video_control,
        control_video,
        start_frame,
        frame_skip,
        control_type,
        ref_images,
        conditioning_scale,
        seed
    )

def copy_settings(job_id):
    """Copies settings from a job to the current tab."""
    (
        input_type,
        prompt,
        num_frames,
        steps,
        flow_shift,
        height,
        width,
        runs,
        source_image,
        source_video,
        frame_image,
        slider_value,
        add_video_control,
        control_video,
        start_frame,
        frame_skip,
        control_type,
        ref_images,
        conditioning_scale,
        seed
    ) = extract_settings_from_job(job_id)

    empty_tab_outputs = [gr.skip()] * 22

    if input_type == "T2V":
        return (
            gr.update(value=prompt),
            gr.update(value=num_frames),
            gr.update(value=steps),
            gr.update(value=flow_shift),
            gr.update(value=height),
            gr.update(value=width),
            gr.update(value=runs),
            gr.skip(),  # No source image for T2V
            gr.skip(),  # No source video for T2V
            gr.skip(),  # No frame image for T2V
            gr.skip(),  # No slider value for T2V
            gr.update(value=add_video_control),
            gr.update(value=control_video),
            gr.update(value=start_frame),
            gr.update(value=frame_skip),
            gr.update(value=control_type),
            *[gr.update(img, visible=True) if img is not None else gr.skip() for img in ref_images],
            gr.update(value=conditioning_scale),
            gr.skip(),  # skip seed

            *empty_tab_outputs,
            *empty_tab_outputs,

            gr.Tabs(selected=0)  # switches tab to T2V
        )
    elif input_type == "I2V":
        return (
            *empty_tab_outputs,

            gr.update(value=prompt),
            gr.update(value=num_frames),
            gr.update(value=steps),
            gr.update(value=flow_shift),
            gr.update(value=height),
            gr.update(value=width),
            gr.update(value=runs),
            gr.update(value=source_image),
            gr.skip(),  # No source video for I2V
            gr.skip(),  # No frame image for I2V
            gr.skip(),  # No slider value for I2V
            gr.update(value=add_video_control),
            gr.update(value=control_video),
            gr.update(value=start_frame),
            gr.update(value=frame_skip),
            gr.update(value=control_type),
            *[gr.update(img, visible=True) if img is not None else gr.skip() for img in ref_images],
            gr.update(value=conditioning_scale),
            gr.skip(),  # skip seed

            *empty_tab_outputs,

            gr.Tabs(selected=1)  # switches tab to I2V
        )
    elif input_type == "LF2V":
        return (
            *empty_tab_outputs,
            *empty_tab_outputs,

            gr.update(value=prompt),
            gr.update(value=num_frames),
            gr.update(value=steps),
            gr.update(value=flow_shift),
            gr.update(value=height),
            gr.update(value=width),
            gr.update(value=runs),
            gr.skip(),  # No source image for LF2V
            gr.update(value=source_video),
            gr.update(value=frame_image),
            gr.update(value=slider_value),
            gr.update(value=add_video_control),
            gr.update(value=control_video),
            gr.update(value=start_frame),
            gr.update(value=frame_skip),
            gr.update(value=control_type),
            *[gr.update(img, visible=True) if img is not None else gr.skip() for img in ref_images],
            gr.update(value=conditioning_scale),
            gr.skip(),  # skip seed

            gr.Tabs(selected=2)  # switches tab to LF2V
        )


def copy_to_lf2v(job_id):
    """Copies settings from a job to the current tab."""
    job_doc = FirestoreFunctions.db.collection('wanVideoJobs').document(job_id).get()

    job_data = job_doc.to_dict()
    user_id = job_data.get("userId")
    result_video = job_data.get("storage_paths", {}).get("result_video")
    result_video = download_result_video(user_id, job_id)  # Ensure the result video is downloaded
    total_frames = get_total_frames(result_video)  # This will ensure the video is processed to get the total frames

    # if there's a control video, increment the start frame by num_frames-1

    (
        input_type,
        prompt,
        num_frames,
        steps,
        flow_shift,
        height,
        width,
        runs,
        source_image,
        source_video,
        frame_image,
        slider_value,
        add_video_control,
        control_video,
        start_frame,
        frame_skip,
        control_type,
        ref_images,
        conditioning_scale,
        seed
    ) = extract_settings_from_job(job_id)

    empty_tab_outputs = [gr.skip()] * 20

    return (
        *empty_tab_outputs,
        *empty_tab_outputs,

        gr.update(value=prompt),
        gr.update(value=num_frames),
        gr.update(value=steps),
        gr.update(value=flow_shift),
        gr.update(value=height),
        gr.update(value=width),
        gr.update(value=runs),
        gr.skip(),  # No source image for LF2V
        gr.update(value=result_video),  # Use the result video as source video
        gr.update(value=frame_image),
        gr.update(value=total_frames - 1),  # Set slider to last frame of the result video
        gr.update(value=add_video_control),
        gr.update(value=control_video),
        gr.update(value=start_frame if not add_video_control else start_frame + num_frames - 1),  # Adjust start frame if control video is present
        gr.update(value=frame_skip),
        gr.update(value=control_type),
        *[gr.update(img, visible=True) if img is not None else gr.skip() for img in ref_images],
        gr.update(value=conditioning_scale),
        gr.skip(),  # skip seed

        gr.Tabs(selected=2)  # switches tab to LF2V
    )


MAX_CARDS = 10


def generate_card_data(job):
    """Generates data for a job card."""

    """
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
    """
    job_id = job.id
    job = job.to_dict()

    queuedTime = job.get('queuedTime')
    # convert timestamp to human readable
    queuedTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(queuedTime))


    job_status = job.get('jobStatus')
    job_progress_info = job.get('job_progress_info', "Processing..." if job_status == "started" else "Waiting...")
    job_progress = str(job.get('job_progress', 0)) + "%"  # Default to 0% if not set
    md = f"`{job.get('input_type')}`"
    if job_status == "started":
        md += f"\n`{job_progress_info}` - `{job_progress}`"
    elif job_status == "queued":
        md += f"\n`Queued` - `{queuedTime}`"

    seed = job.get('seed', -1)
    if seed == -1:
        # a generated seed from the backend
        seed = job.get("random_seed", -1)

    meta_string = f"""
                                    - **Job ID**: `{job_id}`
                                    - **Queued**: {queuedTime}
                                    - **Prompt**: `{job.get("prompt")}`
                                    - **Num Frames**: {job.get('num_frames')}
                                    - **Steps**: {job.get('steps')}
                                    - **Dims**: {job.get('width')}×{job.get('height')}
                                    - **Flow Shift**: {job.get('flow_shift'):.1f}
                                    - **Strength**: {job.get('conditioning_scale', 1.0):.1f}
                                    - **Seed**: {seed}
                                    """

    # Download reference images
    storage_paths = job.get('storage_paths', {})
    ref = storage_paths.get('reference_images', None)
    ref_images = [None for _ in range(4)]  # Initialize with None for up to 4 images
    if ref:
        for idx, img_path in enumerate(ref):
            ref_image_path = download_reference_image(firebase_uid.value, job_id, idx)
            ref_images[idx] = ref_image_path

    add_video_control = job.get('add_video_control')

    if add_video_control:
        meta_string += f"- **Control Type**: {job.get('control_type')}\n"
        meta_string += f"- **Control Frame Index**: {job.get('start_frame')}\n"
        meta_string += f"- **Control Frame Skip**: {job.get('frame_skip')}\n"

        control_video_path = download_control_video(firebase_uid.value, job_id)

    else:
        control_video_path = None

    if job.get('input_type') == 'LF2V':
        meta_string += f"- **Source Frame Index**: {job.get('slider_value')}\n"

        # Download source video for LF2V
        source_video_path = download_source_video(firebase_uid.value, job_id)
    else:
        source_video_path = None

    if job.get('input_type') == 'I2V':
        # Download source image for I2V
        source_image_path = download_source_image(firebase_uid.value, job_id)
    else:
        source_image_path = None

    if job_status == "ended":
        result_video_path = download_result_video(firebase_uid.value, job_id)
    else:
        result_video_path = None

    return md, meta_string, result_video_path, ref_images, source_image_path, source_video_path, control_video_path, job_progress_info, job_progress


def refresh_cards(uid, token, page, *original_job_states, queued_only=False):

    original_job_id_states, original_job_status_states = original_job_states[:MAX_CARDS], original_job_states[MAX_CARDS:]

    jobs = fetch_jobs(uid, token, page, raw=True)
    updates = []

    # pre-create all cards
    card_containers, id_markdowns, videos = ([] for _ in range(3))

    src_img_accs, ref_img_accs, control_vid_accs, source_video_accs = ([] for _ in range(4))
    history_source_images, history_ref_images, history_control_videos, history_source_videos = ([] for _ in range(4))
    meta_accs, metas = ([] for _ in range(2))
    btn_i2v, btn_lf2v = ([] for _ in range(2))

    job_id_states, job_status_states, job_progress_infos, job_progress_bars = ([] for _ in range(4))

    for idx in range(MAX_CARDS):
        visible = idx < len(jobs)
        job = jobs[idx]
        job_status = job.get('jobStatus')
        if idx < len(jobs):

            original_job_id = original_job_id_states[idx]
            original_job_status = original_job_status_states[idx]
            if job.id != original_job_id or job_status != original_job_status or not queued_only or job_status in ('queued', 'started'):
                card_data = generate_card_data(job)
                md, meta_string, result_video_path, ref_image_paths, source_image_path, source_video_path, control_video_path, job_progress_info, job_progress = card_data

                card_containers.append(gr.update(visible=visible))
                id_markdowns.append(gr.update(value=md, visible=visible))
                videos.append(gr.update(value=result_video_path, visible=visible))

                if job.id != original_job_id:

                    ref_img_accs.append(gr.update(open=False, visible=ref_image_paths[0] is not None))
                    history_ref_images.append(gr.update(value=ref_image_paths[0]))

                    src_img_accs.append(gr.update(open=False, visible=source_image_path is not None))
                    history_source_images.append(gr.update(value=source_image_path))

                    source_video_accs.append(gr.update(open=False, visible=source_video_path is not None))
                    history_source_videos.append(gr.update(value=source_video_path))

                    control_vid_accs.append(gr.update(open=False, visible=control_video_path is not None))
                    history_control_videos.append(gr.update(value=control_video_path))

                    meta_accs.append(gr.update(open=False, visible=visible))
                    metas.append(gr.update(value=meta_string, visible=visible))
                else:
                    ref_img_accs.append(gr.skip())
                    history_ref_images.append(gr.skip())
                    src_img_accs.append(gr.skip())
                    history_source_images.append(gr.skip())
                    source_video_accs.append(gr.skip())
                    history_source_videos.append(gr.skip())
                    control_vid_accs.append(gr.skip())
                    history_control_videos.append(gr.skip())
                    meta_accs.append(gr.skip())
                    metas.append(gr.skip())

                btn_i2v.append(gr.update(visible=visible))
                btn_lf2v.append(gr.update(interactive=True, visible=visible))

                job_id_states.append(gr.update(value=job.id))
                job_status_states.append(gr.update(value=job_status))
                # job_progress_infos.append(gr.update(value=job_progress_info, visible=job_status in ('queued', 'started')))
                job_progress_infos.append(gr.update(value=job_progress_info, visible=False))
                # TODO make actual bar
                # job_progress_bars.append(gr.update(value=job_progress, visible=job_status in ('queued', 'started')))
                job_progress_bars.append(gr.update(value=job_progress, visible=False))
            else:
                card_containers.append(gr.skip())
                id_markdowns.append(gr.skip())
                videos.append(gr.skip())
                ref_img_accs.append(gr.skip())
                history_ref_images.append(gr.skip())
                src_img_accs.append(gr.skip())
                history_source_images.append(gr.skip())
                source_video_accs.append(gr.skip())
                history_source_videos.append(gr.skip())
                control_vid_accs.append(gr.skip())
                history_control_videos.append(gr.skip())
                meta_accs.append(gr.skip())
                metas.append(gr.skip())
                btn_i2v.append(gr.skip())
                btn_lf2v.append(gr.skip())

                job_id_states.append(gr.skip())
                job_status_states.append(gr.skip())
                job_progress_infos.append(gr.skip())
                job_progress_bars.append(gr.skip())

            # do not update if queued only
        else:
            card_containers.append(gr.update(visible=False))
            id_markdowns.append(gr.update(value="", visible=False))
            videos.append(gr.update(value=None))

            ref_img_accs.append(gr.update(open=False, visible=False))
            history_ref_images.append(gr.update(value=None))

            src_img_accs.append(gr.update(open=False, visible=False))
            history_source_images.append(gr.update(value=None))

            source_video_accs.append(gr.update(open=False, visible=False))
            history_source_videos.append(gr.update(value=None))

            control_vid_accs.append(gr.update(open=False, visible=False))
            history_control_videos.append(gr.update(value=None))

            meta_accs.append(gr.update(open=False, visible=False))
            metas.append(gr.update(value=""))
            btn_i2v.append(gr.update(visible=False))
            btn_lf2v.append(gr.update(interactive=job_status=="ended", visible=False))

            job_id_states.append(gr.update(value=job.id))
            job_status_states.append(gr.update(job_status))
            job_progress_infos.append(gr.update(visible=False))
            job_progress_bars.append(gr.update(visible=False))


    return (*card_containers, *id_markdowns, *videos,
            *ref_img_accs, *history_ref_images,
            *src_img_accs, *history_source_images,
            *source_video_accs, *history_source_videos,
            *control_vid_accs, *history_control_videos,
            *meta_accs, *metas, *btn_i2v, *btn_lf2v,
            *job_id_states, *job_status_states, *job_progress_infos, *job_progress_bars,
            )


CACHE_DIR = "video_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
def download_result_video(user_id, job_id):
    # 1) Build the storage path
    blob_path = f"users/{user_id}/wanVideoJobs/{job_id}/result_video.mp4"
    # 1) Compute a unique, persistent local path
    local_path = os.path.join(
        CACHE_DIR, f"{user_id}_{job_id}_result_video.mp4"
    )

    FirestoreFunctions.chunked_download(blob_path, local_path)

    return local_path  # gr.File will pick this up and serve it
def download_control_video(user_id, job_id):
    # 1) Build the storage path
    blob_path = f"users/{user_id}/wanVideoJobs/{job_id}/control_video.mp4"
    # 1) Compute a unique, persistent local path
    local_path = os.path.join(
        CACHE_DIR, f"{user_id}_{job_id}_control_video.mp4"
    )

    FirestoreFunctions.chunked_download(blob_path, local_path)

    return local_path  # gr.File will pick this up and serve it
def download_reference_image(user_id, job_id, idx):
    # 1) Build the storage path
    blob_path = f"users/{user_id}/wanVideoJobs/{job_id}/ref_{idx}.png"
    # 1) Compute a unique, persistent local path
    local_path = os.path.join(
        CACHE_DIR, f"{user_id}_{job_id}_ref_{idx}.png"
    )

    FirestoreFunctions.chunked_download(blob_path, local_path)

    return local_path
def download_source_video(user_id, job_id):
    # 1) Build the storage path
    blob_path = f"users/{user_id}/wanVideoJobs/{job_id}/source_video.mp4"
    # 1) Compute a unique, persistent local path
    local_path = os.path.join(
        CACHE_DIR, f"{user_id}_{job_id}_source_video.mp4"
    )

    FirestoreFunctions.chunked_download(blob_path, local_path)

    return local_path
def download_frame_image(user_id, job_id):
    # 1) Build the storage path
    blob_path = f"users/{user_id}/wanVideoJobs/{job_id}/frame_image.png"
    # 1) Compute a unique, persistent local path
    local_path = os.path.join(
        CACHE_DIR, f"{user_id}_{job_id}_frame_image.png"
    )

    FirestoreFunctions.chunked_download(blob_path, local_path)

    return local_path
def download_source_image(user_id, job_id):
    # 1) Build the storage path
    blob_path = f"users/{user_id}/wanVideoJobs/{job_id}/source_image.png"
    # 1) Compute a unique, persistent local path
    local_path = os.path.join(
        CACHE_DIR, f"{user_id}_{job_id}_source_image.png"
    )

    FirestoreFunctions.chunked_download(blob_path, local_path)

    return local_path


def has_active_jobs(firebase_uid, firebase_token):
    # # Authenticate user
    # authenticated, response_message = FirestoreFunctions.authenticate_user(firebase_token, firebase_uid)
    # if not authenticated:
    #     return False

    # Check for active jobs
    query = FirestoreFunctions.db.collection("wanVideoJobs") \
              .where(filter=FieldFilter("userId", "==", firebase_uid)) \
              .where(filter=FieldFilter("jobStatus", "in", ["queued", "started"])).limit(MAX_ACTIVE_JOBS+1) \

    docs = query.get()
    return len(docs)

def poll_job_history(firebase_uid, firebase_token, page, *original_job_states ):
    result_info_message = ""
    active_job_count = has_active_jobs(firebase_uid, firebase_token)

    if active_job_count >= MAX_ACTIVE_JOBS:
        submit_btns = [gr.update(interactive=False) for _ in range(3)]
        result_infos = [gr.update(visible=True, value=result_info_message) for _ in range(3)]
        history_buttons = [gr.update(visible=True) for _ in range(3)]
    else:
        submit_btns = [gr.update(interactive=True) for _ in range(3)]
        result_infos = [gr.update(visible=False) for _ in range(3)]
        history_buttons = [gr.update(visible=False) for _ in range(3)]

    gr_components = update_queued_jobs(firebase_uid, firebase_token, page, *original_job_states)

    return *submit_btns, *result_infos, *history_buttons, gr.update(active=active_job_count>0), *gr_components

def update_queued_jobs(firebase_uid, firebase_token, page, *original_job_states):
    if page != 0:
        return gr.skip()
    gr_components = refresh_cards(firebase_uid, firebase_token, page, *original_job_states,  queued_only=True)
    return gr_components


if __name__ == "__main__":

    with gr.Blocks() as demo:
        # Hidden fields to capture token and uid from URL
        gr.HTML("""
                <script>
                window.addEventListener('load', () => {
                    const params = new URLSearchParams(window.location.search);
                    const token = params.get('token');
                    const uid = params.get('uid');
                    if (token) document.getElementById('id_token').value = token;
                    if (uid) document.getElementById('user_id').value = uid;
                });
                </script>
                """)
        firebase_token = gr.Textbox(label="User Token", visible=False, elem_id='firebase_token')
        firebase_uid = gr.Textbox(label="User ID", visible=False, elem_id='firebase_uid', value="AjS0iUUB8yV2TG8i8Ip48TdNoiM2")

        input_types = ["T2V", "I2V", "LF2V"]
        tab_elements = {
            "T2V": {},
            "I2V": {},
            "LF2V": {},
        }

        # create and return elements for each tab
        copy_outputs = []
        with gr.Tabs() as tabs:
            for input_type_idx, input_type in enumerate(input_types):
                with gr.TabItem(input_type, id=input_type_idx) as tab:
                    block, submit_inputs, submit_btn, source_image, source_video, result_info, history_button = make_tab(input_type)
                tab_elements[input_type]["tab"] = tab
                tab_elements[input_type]["block"] = block

                copy_outputs.extend(submit_inputs)
                tab_elements[input_type]["submit_inputs"] = submit_inputs

                tab_elements[input_type]["submit_btn"] = submit_btn
                tab_elements[input_type]["source_image"] = source_image
                tab_elements[input_type]["source_video"] = source_video
                tab_elements[input_type]["result_info"] = result_info
                tab_elements[input_type]["history_button"] = history_button

            # job history tab
            with gr.TabItem("Job History", id=len(tab_elements)) as job_history_tab:
                with gr.Row():
                    gr.Markdown("### Job History")
                    # Refresh button
                    refresh_btn = gr.Button("Refresh Jobs")

                # pre-create all cards
                card_containers, id_markdowns, videos = ([] for _ in range(3))
                job_id_states, job_status_states, job_progress_infos, job_progress_bars = ([] for _ in range(4))
                src_img_accs, ref_img_accs, control_vid_accs, source_video_accs = ([] for _ in range(4))
                history_source_images, history_ref_images, history_control_videos, history_source_videos = ([] for _ in range(4))
                meta_accs, metas = ([] for _ in range(2))
                btn_i2v, btn_lf2v = ([] for _ in range(2))

                _jobs = fetch_jobs(firebase_uid.value, firebase_token.value, 0, raw=True)
                # Create a grid layout for job cards with columns and rows (2 per row and 5 columns)
                # arrange the indices in a grid-like structure

                for _idx in range(MAX_CARDS//2):

                    with gr.Row():
                        for o_e in range(2):
                            idx = _idx * 2 + o_e
                            meta_string = ""
                            md = ""
                            result_video_path = None
                            visible = idx < len(_jobs)
                            if idx < len(_jobs):
                                job = _jobs[idx]
                                card_data = generate_card_data(job)
                                md, meta_string, result_video_path, ref_image_paths, source_image_path, source_video_path, control_video_path, job_progress_info, job_progress = card_data
                            with gr.Column() as card:
                                # header
                                md = gr.Markdown(md, elem_id=f"job_md_{idx}", visible=visible)

                                job_id = job.id
                                job_id_state = gr.Textbox(job_id, label="Job ID", elem_id=f"job_id_{idx}", interactive=False, visible=False)

                                job_status = job.get('jobStatus')
                                job_status_state = gr.Textbox(job_status, label="Job Status", elem_id=f"job_status_{idx}", interactive=False, visible=False)

                                # job_progress_info = gr.Markdown(job_progress_info, visible=job_status in ("queued", "active"), elem_id=f"job_progress_info_{idx}")
                                job_progress_info = gr.Markdown(job_progress_info, visible=False, elem_id=f"job_progress_info_{idx}")
                                # job_progress_bar = gr.Markdown(job_progress, elem_id=f"job_progress_{idx}", visible=job_status in ("queued", "active"))
                                job_progress_bar = gr.Markdown(job_progress, elem_id=f"job_progress_{idx}", visible=False)

                                # media area
                                history_result_video = gr.Video(result_video_path, label="Result Video", interactive=False, visible=visible, elem_id=f"job_vid_{idx}", height=300, scale=1.0)

                                with gr.Accordion("Reference Image", open=False, visible=ref_image_paths[0] is not None) as ref_img_acc:
                                    history_ref_image = gr.Image(ref_image_paths[0], label="Reference Image", interactive=False, elem_id=f"job_ref_img_{idx}", height=300, scale=1.0)

                                with gr.Accordion("Source Image", open=False, visible=source_image_path is not None) as src_img_acc:
                                    history_source_image = gr.Image(source_image_path, label="Source Image", interactive=False, elem_id=f"job_src_img_{idx}", height=300, scale=1.0)

                                with gr.Accordion("Source Video", open=False, visible=source_video_path is not None) as source_vid_acc:
                                    history_source_video = gr.Video(source_video_path, label="Source Video", interactive=False, elem_id=f"job_src_vid_{idx}", height=300, scale=1.0)

                                with gr.Accordion("Control Video", open=False, visible=control_video_path is not None) as control_vid_acc:
                                    history_control_video = gr.Video(control_video_path, label="Control Video", interactive=False, elem_id=f"job_control_vid_{idx}", height=300, scale=1.0)

                                # accordion for meta
                                with gr.Accordion("Metadata", open=False, visible=visible) as acc:
                                    meta = gr.Markdown(meta_string, elem_id=f"job_meta_{idx}")

                                # action buttons
                                with gr.Row():
                                    b_i2v = gr.Button("Copy to i2v", size="sm", interactive=True, visible=visible)
                                    b_lf2v = gr.Button("Copy to lf2v", size="sm", interactive=job_status=="ended", visible=visible)

                                    b_i2v.click(
                                        copy_settings,
                                        inputs=[job_id_state],
                                        outputs=[
                                            *copy_outputs,
                                            tabs
                                        ]
                                    )

                                    b_lf2v.click(
                                        copy_to_lf2v,
                                        inputs=[job_id_state],
                                        outputs=[
                                            *copy_outputs,
                                            tabs
                                        ]
                                    )

                            card_containers.append(card)
                            id_markdowns.append(md)
                            videos.append(history_result_video)

                            ref_img_accs.append(ref_img_acc)
                            history_ref_images.append(history_ref_image)

                            src_img_accs.append(src_img_acc)
                            history_source_images.append(history_source_image)

                            source_video_accs.append(source_vid_acc)
                            history_source_videos.append(history_source_video)

                            control_vid_accs.append(control_vid_acc)
                            history_control_videos.append(history_control_video)

                            meta_accs.append(acc)
                            metas.append(meta)
                            btn_i2v.append(b_i2v)
                            btn_lf2v.append(b_lf2v)

                            job_id_states.append(job_id_state)
                            job_status_states.append(job_status_state)
                            job_progress_bars.append(job_progress_bar)
                            job_progress_infos.append(job_progress_info)

                # Pagination buttons
                with gr.Row():
                    prev_page_btn = gr.Button("Previous Page")
                    page = gr.Number(label="Page", value=0, precision=0, interactive=True, minimum=0)
                    next_page_btn = gr.Button("Next Page")
                refresh_btn2 = gr.Button("Refresh Jobs")

                outputs=[
                        *card_containers,
                        *id_markdowns,
                        *videos,

                        *ref_img_accs, *history_ref_images,
                        *src_img_accs, *history_source_images,
                        *source_video_accs, *history_source_videos,
                        *control_vid_accs, *history_control_videos,

                        *meta_accs, *metas,
                        *btn_i2v, *btn_lf2v,

                        *job_id_states, *job_status_states, *job_progress_infos, *job_progress_bars,
                    ]

                refresh_btn.click(
                    refresh_cards,
                    inputs=[firebase_uid, firebase_token, page, *(job_id_states + job_status_states)],
                    outputs=outputs
                )

                refresh_btn2.click(
                    refresh_cards,
                    inputs=[firebase_uid, firebase_token, page, *(job_id_states + job_status_states)],
                    outputs=outputs
                )

                prev_page_btn.click(
                    update_page_prev,
                    inputs=[firebase_uid, firebase_token, page],
                    outputs=[page]
                ).then(
                    fn=refresh_cards,
                    inputs=[firebase_uid, firebase_token, page, *(job_id_states + job_status_states)],
                    outputs=outputs
                )

                next_page_btn.click(
                    update_page_next,
                    inputs=[firebase_uid, firebase_token, page],

                    outputs=[page]
                ).then(
                    fn=refresh_cards,
                    inputs=[firebase_uid, firebase_token, page, *(job_id_states + job_status_states)],
                    outputs=outputs
                )

        # gather all result_info, submit button, and history button
        submit_btns, result_infos, history_buttons = ([] for _ in range(len(tab_elements)))
        for input_type in input_types:
            submit_btn = tab_elements[input_type]["submit_btn"]
            result_info = tab_elements[input_type]["result_info"]
            history_button = tab_elements[input_type]["history_button"]

            # “Go to Job History” just switches tabs
            history_button.click(lambda: gr.Tabs(selected=len(input_types)), [], [tabs])

            submit_btns.append(submit_btn)
            result_infos.append(result_info)
            history_buttons.append(history_button)


        # A Timer that runs `poll_controller` every 5 seconds, and once at load
        job_history_poller = gr.Timer(5)

        outputs = [*submit_btns, *result_infos, *history_buttons, job_history_poller]

        job_history_poller.tick(
            fn=poll_job_history,
            inputs=[firebase_uid, firebase_token, page, *(job_id_states + job_status_states)],
            outputs=[*outputs, *card_containers,
                *id_markdowns,
                *videos,

                *ref_img_accs, *history_ref_images,
                *src_img_accs, *history_source_images,
                *source_video_accs, *history_source_videos,
                *control_vid_accs, *history_control_videos,

                *meta_accs, *metas,
                *btn_i2v, *btn_lf2v,

                *job_id_states, *job_status_states, *job_progress_infos, *job_progress_bars,
            ]
        )

        # Wire up the submit buttons for each tab
        for input_type in input_types:
            submit_btn = tab_elements[input_type]["submit_btn"]
            submit_inputs = tab_elements[input_type]["submit_inputs"]

            # insert the firebase token and uid into the submit inputs
            submit_inputs.insert(0, firebase_token)
            submit_inputs.insert(1, firebase_uid)

            # Set the submit button to call the appropriate function based on input type
            if input_type == "I2V":

                submit_btn.click(
                    on_submit_i2v,
                    inputs=submit_inputs,
                    outputs=outputs
                )
            elif input_type == "LF2V":
                submit_btn.click(
                    on_submit_lf2v,
                    inputs=submit_inputs,
                    outputs=outputs
                )
            else:
                submit_btn.click(
                    on_submit_t2v,
                    inputs=submit_inputs,
                    outputs=outputs
                )

    demo.title = "Video Generator and Editor"

    # Set the width of the main container to 900px
    demo.css = """

        .gradio-container {
            max-width: 1200px;
        }
        """

    demo.launch()