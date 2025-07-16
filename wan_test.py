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
from dfs_seg_inpaint_swap.redresser_utils import RedresserSettings
from dfs_seg_inpaint_swap.wan import VideoAnnotator, WanVideoGenerator, WanSettings
from dfs_seg_inpaint_swap.fire_functions import FirestoreFunctions
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import time
import uuid
import os
from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

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


def update_slider_from_video(video_path, current_time):
    # Estimate frame index from current_time (in seconds)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    frame_idx = int(current_time * fps)
    return frame_idx


def on_slider_change(video_path, frame_idx, current_time):
    # Calculate the time for the desired frame
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    target_time = frame_idx / fps if fps else 0

    # Only update if the time is different (avoid double dipping)
    if abs(target_time - current_time) > (1.0 / max(fps, 1)) / 2:
        return gr.update(playback_position=target_time)
    return gr.update()


def make_source_video_slider():

    source_video = gr.Video(label="Source Video", )

    frame_image = gr.Image(label="Source Frame", type="filepath")
    slider = gr.Slider(
        minimum=0,
        maximum=1,
        step=1,
        value=0,
        label="Frame Index",
        interactive=True,
        show_label=True,
        elem_id="source_frame_slider",
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
    prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here")
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
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here",)
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
                    height = gr.Number(label="Height", value=480, precision=0, interactive=True, minimum=240,
                                       maximum=896, step=16)
                    width = gr.Number(label="Width", value=832, precision=0, interactive=True, minimum=240, maximum=896,
                                      step=16)

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
                source_image = None
                source_video = None
                slider = None
                if input_type == "I2V":
                    gr.Markdown("### Image Source Options (Select an image to generate from)")
                    source_image = gr.Image(label="Source Image", type="filepath")
                elif input_type == "LF2V":
                    gr.Markdown(
                        "### Video Source Options (Select a video, then select a frame to generate/continue from)")
                    source_video, frame_image, slider = make_source_video_slider()

                # Height and Width for I2V and LF2V
                if input_type in ["I2V", "LF2V"]:

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
                    with gr.Column():
                        start_frame = gr.Number(label="Start Control Frame", value=0, precision=0, interactive=True,
                                                minimum=0)
                        frame_skip = gr.Number(label="Frame Skip", value=0, precision=0, interactive=True, minimum=0,
                                               maximum=3)
                        control_type = gr.Dropdown(
                            label="Control Type",
                            choices=["Full Pose", "Person Mask", "Clothing Mask", "Face Mask", "Background Mask",
                                     "Depth Mask"],
                            value="Full Pose",
                            interactive=True
                        )
                add_video_control.change(
                    lambda v: gr.update(visible=v),
                    inputs=add_video_control,
                    outputs=video_control_options
                )

        # --- Submit Button ---
        runs = gr.Number(label="Runs", value=1, precision=0, interactive=True, minimum=1, maximum=10)
        submit_btn = gr.Button("Submit", variant="primary")

        # result video output
        with gr.Row():
            with gr.Column():
                result_video_clip = gr.Video(label="Result Video Clip", interactive=False, visible=False)
                copy_clip_to_lf2v_btn = gr.Button("Copy to LF2V", visible=False)
            with gr.Column():
                result_video_full = gr.Video(label="Result Video Full", interactive=False, visible=False)
                copy_full_to_lf2v_btn = gr.Button("Copy to LF2V", visible=False)
        result_info = gr.Label("Generation completed in -", visible=False)

        # Prepare input list for submit button
        submit_inputs = [prompt, num_frames, steps, flow_shift, height, width, runs]

        if input_type == "I2V":
            submit_inputs.extend([source_image])
            submit_inputs.extend([
                add_video_control, control_video, start_frame, frame_skip, control_type, *ref_image_boxes
            ])

        elif input_type == "LF2V":
            submit_inputs.extend([source_video, frame_image, slider])
            submit_inputs.extend([
                add_video_control, control_video, start_frame, frame_skip, control_type, *ref_image_boxes
            ])

        else:
            submit_inputs.extend([
                add_video_control, control_video, start_frame, frame_skip, control_type, *ref_image_boxes
            ])

    return block, submit_inputs, submit_btn, source_image, source_video, result_info, result_video_clip, result_video_full, copy_clip_to_lf2v_btn, copy_full_to_lf2v_btn


def on_submit_t2v(
        firebase_token, firebase_uid, prompt, num_frames, steps, flow_shift, height, width, runs,
        add_video_control=False, control_video=None, start_frame=0, frame_skip=0, control_type="Full Pose",
        *ref_image_boxes,
):
    # ignore source image and video inputs for T2V
    return on_submit(
        firebase_token, firebase_uid, "T2V", prompt, num_frames, steps, flow_shift, height, width, runs,
        source_image=None, source_video=None, frame_image=None, slider_value=None,
        add_video_control=add_video_control, control_video=control_video, start_frame=start_frame,
        frame_skip=frame_skip, control_type=control_type, ref=ref_image_boxes,
    )


def on_submit_i2v(
        firebase_token, firebase_uid, prompt, num_frames, steps, flow_shift, height, width, runs,
        source_image=None,
        add_video_control=False, control_video=None, start_frame=0, frame_skip=0, control_type="Full Pose",
        *ref_image_boxes,
):
    return on_submit(
        firebase_token, firebase_uid, "I2V", prompt, num_frames, steps, flow_shift, height, width, runs,
        source_image=source_image, source_video=None, frame_image=None, slider_value=None,
        add_video_control=add_video_control, control_video=control_video, start_frame=start_frame,
        frame_skip=frame_skip, control_type=control_type, ref=ref_image_boxes,
    )


def on_submit_lf2v(
        firebase_token, firebase_uid, prompt, num_frames, steps, flow_shift, height, width, runs,
        source_video=None, frame_image=None, slider_value=None,
        add_video_control=False, control_video=None, start_frame=0, frame_skip=0, control_type="Full Pose",
        *ref_image_boxes,
):
    return on_submit(
        firebase_token, firebase_uid, "LF2V", prompt, num_frames, steps, flow_shift, height, width, runs,
        source_image=None, source_video=source_video, frame_image=frame_image, slider_value=slider_value,
        add_video_control=add_video_control, control_video=control_video, start_frame=start_frame,
        frame_skip=frame_skip, control_type=control_type, ref=ref_image_boxes,
    )


def on_submit(
        firebase_token, firebase_uid, input_type, prompt, num_frames, steps, flow_shift, height, width, runs,
        source_image=None, source_video=None, frame_image=None, slider_value=None,
        add_video_control=False, control_video=None, start_frame=0, frame_skip=0, control_type="Full Pose", ref=None, progress=gr.Progress(track_tqdm=True)
):
    start_time = time.time()

    authenticated, response_message = FirestoreFunctions.authenticate_user(firebase_token, firebase_uid)
    # if not authenticated:
    #     return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=response_message)
    # else:
    #     submit_to_firebase()

    submit_to_firebase(firebase_uid, input_type, prompt, num_frames, steps, flow_shift, height, width, runs,
        source_image=source_image, source_video=source_video, frame_image=frame_image, slider_value=slider_value,
        add_video_control=add_video_control, control_video=control_video, start_frame=start_frame, frame_skip=frame_skip, control_type=control_type, ref=ref)
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=response_message)


def submit_to_firebase(
        firebase_uid, input_type, prompt, num_frames, steps, flow_shift, height, width, runs,
        source_image=None, source_video=None, frame_image=None, slider_value=None,
        add_video_control=False, control_video=None, start_frame=0, frame_skip=0, control_type="Full Pose", ref=None
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
        'queuedTime': time.time()
    }
    doc_ref.set(doc_data)
    print("Firestore document created with ID:", job_id)
    print("Document data:", doc_data)


PAGE_SIZE = 10

def fetch_jobs(firebase_uid, firebase_token, page):

    # Authenticate user
    # authenticated, response_message = FirestoreFunctions.authenticate_user(firebase_token, firebase_uid)

    query = FirestoreFunctions.db.collection("wanVideoJobs") \
              .where(filter=FieldFilter("userId", "==", firebase_uid))

    query = query.order_by("queuedTime", direction=firestore.firestore.Query.DESCENDING) \
                 .offset(page * PAGE_SIZE).limit(PAGE_SIZE)

    docs = query.get()
    rows = []
    for doc in docs:
        data = doc.to_dict()
        rows.append([
            doc.id,
            data.get("jobStatus", ""),
            data.get("prompt", ""),
            int(data.get("queuedTime", 0)) if data.get("queuedTime") else 0,
            int(data.get("startedTime", 0)) if data.get("startedTime") else 0,
            int(data.get("endedTime", 0)) if data.get("endedTime") else 0
        ])
    print(f"Fetched {len(rows)} jobs for page {page}.")
    print("Rows:", rows)
    return rows

def refresh_jobs(firebase_uid, firebase_token, page):
    # Authenticate user
    # authenticated, response_message = FirestoreFunctions.authenticate_user(firebase_token, firebase_uid)
    # if not authenticated:
    #     return gr.update(value=response_message)

    # Fetch jobs for the current page
    jobs = fetch_jobs(firebase_uid, firebase_token, page)
    return gr.update(value=jobs)


def update_page(firebase_uid, firebase_token, page, direction):
    new_page = max(0, page + direction)
    jobs = fetch_jobs(firebase_uid, firebase_token, new_page)
    return gr.update(value=jobs), gr.update(value=new_page)


def update_page_prev(firebase_uid, firebase_token, page):
    return update_page(firebase_uid, firebase_token, page, -1)


def update_page_next(firebase_uid, firebase_token, page):
    return update_page(firebase_uid, firebase_token, page, 1)


def copy_to_lf2v(result_video_path):
    # This function will be called when the button is clicked
    # It returns the video path to the LF2V source_video
    return gr.update(value=result_video_path), gr.update(selected="LF2V")


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
        with gr.Tabs() as tabs:
            for input_type in input_types:
                with gr.TabItem(input_type) as tab:
                    block, submit_inputs, submit_btn, source_image, source_video, result_info, result_video_clip, result_video_full, copy_clip_to_lf2v_btn, copy_full_to_lf2v_btn = make_tab(input_type)
                tab_elements[input_type]["tab"] = tab
                tab_elements[input_type]["block"] = block
                tab_elements[input_type]["submit_inputs"] = submit_inputs
                tab_elements[input_type]["submit_btn"] = submit_btn
                tab_elements[input_type]["source_image"] = source_image
                tab_elements[input_type]["source_video"] = source_video
                tab_elements[input_type]["result_info"] = result_info
                tab_elements[input_type]["result_video_clip"] = result_video_clip
                tab_elements[input_type]["result_video_full"] = result_video_full
                tab_elements[input_type]["copy_clip_to_lf2v_btn"] = copy_clip_to_lf2v_btn
                tab_elements[input_type]["copy_full_to_lf2v_btn"] = copy_full_to_lf2v_btn

            # job history tab
            with gr.TabItem("Job History") as job_history_tab:
                gr.Markdown("### Job History")

                page = gr.Number(label="Page", value=0, precision=0, interactive=True, minimum=0)
                # Fetch initial jobs
                jobs = fetch_jobs(firebase_uid.value, firebase_token.value, page.value)

                # Create a DataFrame to display jobs
                job_table = gr.DataFrame(
                    headers=["Job ID", "Status", "Prompt", "Queued Time", "Started Time", "Ended Time"],
                    datatype=["str", "str", "str", "number", "number", "number"],
                    row_count=10,
                    interactive=True,
                    value=jobs
                )

                # Refresh button
                refresh_btn = gr.Button("Refresh Jobs")
                refresh_btn.click(
                    refresh_jobs,
                    inputs=[firebase_uid, firebase_token, page],
                    outputs=job_table
                )

                # Pagination buttons
                with gr.Row():
                    prev_page_btn = gr.Button("Previous Page")
                    next_page_btn = gr.Button("Next Page")

                prev_page_btn.click(
                    update_page_prev,
                    inputs=[firebase_uid, firebase_token, page],
                    outputs=[job_table, page]
                )
                next_page_btn.click(
                    update_page_next,
                    inputs=[firebase_uid, firebase_token, page],
                    outputs=[job_table, page]
                )

        # Wire up the submit buttons for each tab
        for input_type in input_types:
            submit_btn = tab_elements[input_type]["submit_btn"]
            submit_inputs = tab_elements[input_type]["submit_inputs"]
            result_info = tab_elements[input_type]["result_info"]
            result_video_clip = tab_elements[input_type]["result_video_clip"]
            result_video_full = tab_elements[input_type]["result_video_full"]
            copy_clip_to_lf2v_btn = tab_elements[input_type]["copy_clip_to_lf2v_btn"]
            copy_full_to_lf2v_btn = tab_elements[input_type]["copy_full_to_lf2v_btn"]

            # insert the firebase token and uid into the submit inputs
            submit_inputs.insert(0, firebase_token)
            submit_inputs.insert(1, firebase_uid)
            outputs = [result_video_clip, copy_clip_to_lf2v_btn,
                        result_video_full, copy_full_to_lf2v_btn,
                        result_info]
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

            # Wire up the button to update the LF2V source video and switch tab
            copy_clip_to_lf2v_btn.click(
                copy_to_lf2v,
                inputs=result_video_clip,
                outputs=[tab_elements["LF2V"]["source_video"], tabs]
            )
            copy_full_to_lf2v_btn.click(
                copy_to_lf2v,
                inputs=result_video_full,
                outputs=[tab_elements["LF2V"]["source_video"], tabs]
            )
    demo.title = "Video Generator and Editor"

    # Set the width of the main container to 900px
    demo.css = """

        .gradio-container {
            max-width: 1200px;
        }
        """

    demo.launch()