import datetime
import os
import time
from multiprocessing import Queue, Process

from cog import CogSettings, VideoGenerator
from redresser_utils import SocketServer, SocketClient
from redresser_flux import Redresser, ImageGenerator
from redresser_sd15 import RedresserSD15, ImageGeneratorSD15
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
    def download_image(self, job_doc):
        '''

        :param job_doc:
        :return: str: path to downloaded image
        '''
        self.job_doc = job_doc
        self.job_id = job_doc.id
        self.userId = self.job_doc.get("userId")
        # self.faceModelId = self.job_doc.get("faceModelId")
        #
        # self.model_doc_ref = FirestoreFunctions.usersRef.document(self.userId).collection("faceModels").document(
        #     self.faceModelId)
        # model_doc = self.model_doc_ref.get()
        # if not model_doc.exists:
        #     if self.job_type == FirestoreFunctions.SWAP_IMAGE_JOB:
        #         self.fuck_this_user_image(reset_model_iterations=False, notification_msg_to_user="Model does not exist")
        #         return False

        # set up job paths and remove anything in {self.userModelDir = os.path.join(self.userModelRootDir, self.faceModelId)}
        # make the dir for current user id
        self.userDir = os.path.join(RepaintJobProcesser.USERS_DIR, self.userId)
        if not os.path.exists(self.userDir):
            os.mkdir(self.userDir)

        # make the dir for current user id repainter root
        self.userRepaintRootDir = os.path.join(self.userDir, "repaint")
        if not os.path.exists(self.userRepaintRootDir):
            os.mkdir(self.userRepaintRootDir)

        # make the dir for current user id repainter root
        self.userRepaintDir = os.path.join(self.userRepaintRootDir, self.job_id)
        if not os.path.exists(self.userRepaintDir):
            os.mkdir(self.userRepaintDir)

        self.userRepaintStorageRef = f"users/{self.userId}/redresserImageJobs/{self.job_doc.id}"

        # need to have default for a non repaint job so the pipe knows where to save the image
        self.imageFileName = job_doc.get("imageFileName")
        firebase_storage_path = self.userRepaintStorageRef + "/" + self.imageFileName
        # download the video
        self.input_file_path = self.userRepaintDir + "/" + self.imageFileName
        self.mode = job_doc.get("mode")  # 0 for t2i, 1 for repaint
        if self.mode == 0:
            return self.input_file_path, None

        if not FirestoreFunctions.chunked_download(
                firebase_storage_path=firebase_storage_path,
                download_destination_path=self.input_file_path
        ):
            return False

        #download the mask if is a custom mask
        self.maskOption = job_doc.get("maskOption")
        if self.maskOption == 1:
            self.maskFileName = job_doc.get("maskFileName")
            firebase_storage_path = self.userRepaintStorageRef + "/" + self.maskFileName
            # download the video
            self.input_mask_file_path = self.userRepaintDir + "/" + self.maskFileName
            if not FirestoreFunctions.chunked_download(
                    firebase_storage_path=firebase_storage_path,
                    download_destination_path=self.input_mask_file_path
            ):
                return False
            return self.input_file_path, self.input_mask_file_path
        else:
            return self.input_file_path, None

        # # make the dir for current user id model root
        # self.userModelRootDir = os.path.join(self.userDir, "faceModels")
        # if not os.path.exists(self.userModelRootDir):
        #     os.mkdir(self.userModelRootDir)
        #
        # # make the dir for the current user's face model id
        # self.userModelDir = os.path.join(self.userModelRootDir, self.faceModelId)
        # if clean:
        #     self.clean_up_job()
        # if not os.path.exists(self.userModelDir):
        #     os.mkdir(self.userModelDir)

        # # set the storage ref for checkpoints, face set zips and all other job specific files
        # self.userModelStorageRef = f"users/{self.userId}/faceModels/{self.faceModelId}"

    def complete_job(self):
        '''
        uploads image to redresserImageJobs
        :return:
        '''
        secs = time.time()
        if not FirestoreFunctions.send_job_file(
                local_file_full_path=self.input_file_path.replace(self.imageFileName, "outputImage.png"),
                file_name=f"{self.imageFileName}_repainted.png",
                firebase_storage_path=f"users/{self.userId}/redresserImageJobs/{self.job_id}",
                required=True
        ):
            raise ValueError("SWAPPED IMG DOESN'T EXIST")

        print(f"sending image took {(time.time() - secs):.2f} secs")
        while True:
            try:
                FirestoreFunctions.repaintImageJobsRef.document(self.job_id).update({
                    'jobStatus': "ended",
                    'endedTime': int(time.time()),
                })
                break
            except:
                print("Failed to update swapping doc jobStatus. retrying in 5 seconds...")
                time.sleep(5)


def redresser_flux_process(r, is_server, r_input_queue, r_output_queue):
    pipe_map = {"flux": 0, "flux-fill": 1, "sd15": 2, "sd15-fill": 3}
    if 'fill' in r:

        img_client = SocketClient(5000 + pipe_map[r])
        pipe_server = SocketServer(5100 + pipe_map[r])
    else:
        img_client = None
        pipe_server = None
    pipeline = None

    while True:
        # get parsed options from dfs
        try:
            options = r_input_queue.get(block=True)
        except:
            time.sleep(2)
            continue
        # load pipeline here to test imgage processor
        if pipeline is None:
            pipeline = get_pipeline(r, is_server)

        result = run_redresser_flux_process(
            pipeline=pipeline, options=options,
            img_client=img_client, pipe_server=pipe_server)

        r_output_queue.put(result, block=True)


def get_pipeline(r, is_server):
    _r = r
    if r == "flux-fill":
        r = Redresser(is_server=is_server)
    elif r == "sd15-fill":
        r = RedresserSD15(is_server=is_server)
    elif r == "flux":
        r = ImageGenerator(is_server=is_server)
    elif r == "sd15":
        r = ImageGeneratorSD15(is_server=is_server)
    elif r == "cog-i2v":
        r = VideoGenerator(is_server=is_server)
    return r


def run_redresser_flux_process(pipeline, options, pipe_server:SocketServer, img_client:SocketClient):
    # determine which pipeline to load
    if pipeline.is_server:
        pipeline.settings.map_dfs_options(options)
        if pipeline.settings.options["num_inference_steps"] != 8:
            print("setting num_inference_steps to 8 for turbo")
            pipeline.settings.options["num_inference_steps"] = 8
        if 1.5 > pipeline.settings.options["guidance_scale"] or 5.5 < pipeline.settings.options["guidance_scale"]:
            print("setting guidance_scale to 3.5 for turbo")
            pipeline.settings.options["guidance_scale"] = 3.5
    else:
        pipeline.settings.options = options.copy()
        # need absolute path for the input image
        image_file_path: str = pipeline.settings.options["image"]
        if not image_file_path.startswith("C:"):
            absolute_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                image_file_path)
            pipeline.settings.options["image"] = absolute_path
        print("passed settings", pipeline.settings.options)

    # pass options to image processor
    if isinstance(pipeline, Redresser):
        print("passing settings to image processor")
        while True:
            try:
                img_client.put(pipeline.settings)
                break
            except ConnectionRefusedError:
                print("\rError! Trying again in 5 seconds...", end="")

        # go through each image in dir to wait
        if os.path.isdir(pipeline.settings.options['image']):
            image_dir = pipeline.settings.options['image']
            for file_index, file in enumerate(os.listdir(image_dir)):
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".webp") or file.endswith(".jfif"):
                    print(f"[{file_index}] waiting for image processor outputs for file {file}")
                    im_outputs = pipe_server.get()

                    if im_outputs is None:
                        return False

                    print(f"[{file_index}] parsing image processor outputs for pipe")
                    parsed_im_outputs = pipeline.parse_image_processor_outputs(*im_outputs)

                    print(f"[{file_index}] running pipeline")
                    pipeline.run(*parsed_im_outputs)
        else:
            print("waiting for image processor outputs")
            im_outputs = pipe_server.get()

            if im_outputs is None:
                return False

            print("parsing image processor outputs for pipe")
            parsed_im_outputs = pipeline.parse_image_processor_outputs(*im_outputs)

            print("running pipeline")
            pipeline.run(*parsed_im_outputs)
    else:
        print("running pipeline")
        pipeline.run()

    return True


def run(r="flux", is_server=True):
    pipe_ids = ("flux", "flux-fill", "sd15", "sd15-fill", "cog-i2v")
    if r not in pipe_ids:
        raise ValueError("r must be in one of pipe_ids:", pipe_ids)

    r_input_queue = Queue(1)
    r_output_queue = Queue(1)

    r_process = Process(
        target=redresser_flux_process,
        kwargs={
            "r": r,
            "is_server": is_server,
            "r_input_queue": r_input_queue,
            "r_output_queue": r_output_queue,})

    r_process.start()

    if is_server:
        machine_id = f'OVERLORD4-0'  # use -3 or -0 because -2 is weird

        RepaintJobProcesser.make_dirs(JOB_DIR)
        firestoreFunctions = FirestoreFunctions()
        job_processor = RepaintJobProcesser()

        job_order = 0
        job_orders = {
            0: FirestoreFunctions.REPAINT_IMAGE_JOB,
            # 0: FirestoreFunctions.TRAINING_JOB,
            1: FirestoreFunctions.REPAINT_IMAGE_JOB,
        }

        while True:
            total_secs = time.time()
            secs = time.time()

            job_order = job_order + 1 if job_order < len(job_orders) - 1 else 0
            job_type = job_orders[job_order]

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
                    time.sleep(1)
                    continue
                else:
                    job_to_lock = firestoreFunctions.get_jobs(job_type=job_type, resolutions=[])
                    if job_to_lock is not None:
                        print(f'\nlocking job({job_to_lock.id}) ({job_type})...')
                        firestoreFunctions.lock_job(job_type=job_type, job=job_to_lock)
                        print(f'job locked! keeping same job type ({job_type}) for next loop, sleeping 1 second...')
                        job_order -= 1
                        time.sleep(1)
                        continue
                    else:
                        # Get the current date and time
                        current_time = datetime.datetime.now()

                        # Format and display the current time with AM/PM
                        formatted_time = current_time.strftime("%I:%M:%S %p")
                        print(f'\r[{formatted_time}]No jobs available for ({job_type}), sleeping 5 second...', end="")
                        time.sleep(5)
                        continue

            # download the file to the appropriate folder
            image_path, mask_path = job_processor.download_image(started_job)
            # set the downloaded image path
            job_dict = started_job.to_dict()
            options = job_dict.copy()
            options["image"] = image_path

            if mask_path is not None:
                options["mask"] = mask_path
            else:
                try:
                    options.pop("mask")
                except KeyError:
                    pass

            print("options", options)
            # print("image_path", image_path)
            # run the pipeline with options inputs and runs the image processor if needed
            r_input_queue.put(options)

            result = r_output_queue.get(block=True)

            if not result:
                print("job failed?")
                print("sleeping 5...")
                time.sleep(5)
            else:
                job_processor.complete_job()
                print("job completed")
                print("sleeping 5...")
                time.sleep(5)
    else:
        if r == "cog-i2v":
            settings = CogSettings()
        else:
            settings = RedresserSettings()
        while True:
            settings.set_options()
            runs = int(settings.options.get("runs", 1))
            for _ in range(runs):
                r_input_queue.put(settings.options.copy())

                result = r_output_queue.get(block=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--r", default='flux')
    parser.add_argument("-s", "--is_server", default=False, action='store_true')

    args = parser.parse_args()
    r = args.r
    is_server = args.is_server
    pipe_ids = ("flux", "flux-fill", "sd15", "sd15-fill", "cog-i2v")
    if r not in pipe_ids:
        raise ValueError("r must be in one of pipe_ids:", pipe_ids)
    print(f"running with {r},is_server={is_server}")
    # run("cog-i2v", is_server=False)
    run(r, is_server=is_server)
