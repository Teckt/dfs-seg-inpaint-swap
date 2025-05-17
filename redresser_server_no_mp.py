import datetime
import os
import sys
import time

from cog import CogSettings, VideoGenerator
from wan import WanSettings, WanVideoGenerator
from redresser_utils import SocketServer, SocketClient
from redresser_flux import Redresser
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
            # nothing to download, just return
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

    def complete_job(self, job_dict):
        '''
        uploads image to redresserImageJobs
        :return:
        '''
        secs = time.time()
        # 0 for t2i, 1 for repaint
        if self.mode == 1:
            if not FirestoreFunctions.send_job_file(
                    local_file_full_path=self.input_file_path.replace(self.imageFileName, OUTPUT_ORIGINAL_FILE_BASE_NAME),
                    file_name=f"{self.imageFileName}_original.png",
                    firebase_storage_path=f"users/{self.userId}/redresserImageJobs/{self.job_id}",
                    required=True
            ):
                raise ValueError("SWAPPED IMG DOESN'T EXIST")

            if not FirestoreFunctions.send_job_file(
                    local_file_full_path=self.input_file_path.replace(self.imageFileName, OUTPUT_MASK_FILE_BASE_NAME),
                    file_name=f"{self.imageFileName}_mask.png",
                    firebase_storage_path=f"users/{self.userId}/redresserImageJobs/{self.job_id}",
                    required=True
            ):
                raise ValueError("SWAPPED MASK IMG DOESN'T EXIST")

        if not FirestoreFunctions.send_job_file(
                local_file_full_path=self.input_file_path.replace(self.imageFileName, OUTPUT_FILE_BASE_NAME),
                file_name=f"{self.imageFileName}_repainted.png",
                firebase_storage_path=f"users/{self.userId}/redresserImageJobs/{self.job_id}",
                required=True
        ):
            raise ValueError("SWAPPED IMG DOESN'T EXIST")

        maybe_create_faceswap_job(job_dict=job_dict)

        print(f"sending images took {(time.time() - secs):.2f} secs")
        while True:
            try:
                FirestoreFunctions.repaintImageJobsRef.document(self.job_id).update({
                    'jobStatus': "ended",
                    'endedTime': int(time.time()),
                })
                break
            except Exception as e:
                print(e, "Failed to update swapping doc jobStatus. retrying in 5 seconds...")
                time.sleep(5)


def get_pipeline(r, is_server, use_hyper):
    _r = r
    if r == "flux-fill":
        r = Redresser(is_server=is_server, model="fill", use_hyper=use_hyper)
    elif r == "sd15-fill":
        r = RedresserSD15(is_server=is_server)
    elif r == "flux":
        r = Redresser(is_server=is_server, model="t2i", use_hyper=use_hyper)
    elif r == "sd15":
        r = ImageGeneratorSD15(is_server=is_server)
    elif r == "cog-i2v":
        r = VideoGenerator(is_server=is_server)
    elif r == "wan-480":
        r = WanVideoGenerator(is_server=is_server)
    return r


def run_redresser_flux_process(pipeline, options, pipe_server:SocketServer, img_client:SocketClient, job_processor):
    # 0 for t2i, 1 for repaint
    mode = options.get("mode")
    if pipeline.is_server:
        # insert loras if not ads
        insert_loras = not options.get("createdFromAds", False) and not options.get("useHyper", False)
        pipeline.settings.map_dfs_options(options, mode, insert_loras=insert_loras)
        if not isinstance(pipeline.settings, CogSettings) and not isinstance(pipeline.settings, WanSettings):
            # all models should be fused with either hyper or turbo so keep this at 8
            # if pipeline.settings.options["num_inference_steps"] != 8:
            #     print("setting num_inference_steps to 8 for turbo")
            #     pipeline.settings.options["num_inference_steps"] = 8
            pass

        print("mapped settings", pipeline.settings.options)

    else:
        # TODO need to do something about the newly added 'mode' in options for local
        pipeline.settings.options = options.copy()
        # need absolute path for the input image
        image_file_path: str = pipeline.settings.options["image"]
        if not image_file_path.startswith("C:"):
            absolute_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                image_file_path)
            pipeline.settings.options["image"] = absolute_path
        # forget using masks on local, use on server instead
        pipeline.settings.options["mask"] = None
        print("passed settings", pipeline.settings.options)

    if mode == 1:  # 1 for repaint
        print("passing settings to image processor")
        while True:
            try:
                img_client.put((None if job_processor is None else job_processor.job_id, pipeline.settings))
                break
            except ConnectionRefusedError:
                print("\rError! Trying again in 5 seconds...", end="")

        # go through each image in dir to wait LOCAL ONLY
        if os.path.isdir(pipeline.settings.options['image']):
            image_dir = pipeline.settings.options['image']
            for file_index, file in enumerate(os.listdir(image_dir)):
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".webp") or file.endswith(".jfif"):
                    print(f"[{file_index}] waiting for image processor outputs for file {file}")
                    pipe_output = pipe_server.get()

                    if pipe_output is None:
                        return False
                    # make sure the job id is the same?
                    job_id, im_outputs = pipe_output

                    print(f"[{file_index}] parsing image processor outputs for pipe")
                    parsed_im_outputs = pipeline.parse_image_processor_outputs(*im_outputs)

                    print(f"[{file_index}] running pipeline")
                    pipeline.run(*parsed_im_outputs)
        else:
            print("waiting for image processor outputs")
            pipe_output = pipe_server.get()

            if pipe_output is None:
                return False

            job_id, im_outputs = pipe_output
            # loop until the job id is the same
            if job_processor is not None and job_id != job_processor.job_id:
                print("passing settings to image processor")
                while True:
                    while True:
                        try:
                            img_client.put(None if job_processor is None else job_processor.job_id, pipeline.settings)
                            break
                        except ConnectionRefusedError:
                            print("\rError! Trying again in 1 seconds...", end="")
                            time.sleep(1)
                    print("waiting for image processor outputs")
                    pipe_output = pipe_server.get()

                    if pipe_output is None:
                        return False
                    # make sure the job id is the same?
                    job_id, im_outputs = pipe_output
                    if job_id == job_processor.job_id:
                        break

            print("parsing image processor outputs for pipe")
            parsed_im_outputs = pipeline.parse_image_processor_outputs(*im_outputs)

            print("running pipeline")
            pipeline.run(*parsed_im_outputs)
    else:  # 0 for t2i
        print("running pipeline")
        pipeline.run(None, None, None, None, None, None, None, None)

    return True


def maybe_switch_pipelines(pipeline_switch_server, pipeline):
    # print("maybe_switch_pipelines")
    try:
        to_switch = pipeline_switch_server.get(blocking=False)
        if to_switch == "fill":
            if pipeline.model != "fill":
                print("switching to fill")
                pipeline.switch_pipeline("fill")
        else:
            if pipeline.model == "fill":
                print("switching to t2i")
                pipeline.switch_pipeline("t2i")

    except BlockingIOError:
        pass


def maybe_create_faceswap_job(job_dict: dict):
    """
    checks for the face model name and id
    :param job_dict: redresserImageJob dict
    :return:
    """
    print(job_dict)
    faceModelId = job_dict["faceModelId"]
    if faceModelId == "":
        return

    # create doc in repainterImageFaceswapJob with same id
    FirestoreFunctions.repaintImageFaceswapJobsRef.document(job_dict["id"]).set(
        {
            "userId": job_dict["userId"],
            "jobStatus": "queued",
            "faceModelId": job_dict["faceModelId"],
            "queuedTime": int(time.time()),
            "imageFileName": job_dict["imageFileName"]
        }
    )

    # wait
    while True:
        doc = FirestoreFunctions.repaintImageFaceswapJobsRef.document(job_dict["id"]).get()
        if str(doc.get("jobStatus")) in ["ended", "failed"]:
            return

        print(doc.to_dict(), "sleeping 5...")
        time.sleep(5)


def run(r="flux", is_server=True, machine_id="OVERLORD4-0", use_hyper=False, socket_port=5000):
    
    pipe_map = {"flux": 0, "flux-fill": 0, "sd15": 1, "sd15-fill": 1}
    # if 'fill' in r:
    if r in pipe_map.keys():
        img_client = SocketClient(socket_port + pipe_map[r])
        pipe_server = SocketServer(socket_port + 100 + pipe_map[r])
    else:
        img_client = None
        pipe_server = None
    # else:
    #     img_client = None
    #     pipe_server = None
    # pipeline = None

    # pipeline_switch_server = None  # was testing pipeline switch; should be unnecessary now
    pipeline_switch_server = SocketServer(7777)
    pipeline = get_pipeline(r, is_server, use_hyper)

    if is_server:

        RepaintJobProcesser.make_dirs(JOB_DIR)
        firestoreFunctions = FirestoreFunctions()
        FirestoreFunctions.machine_id = machine_id
        firestoreFunctions.machine_id = machine_id
        job_processor = RepaintJobProcesser()

        job_order = 0
        job_orders = {
            0: FirestoreFunctions.REPAINT_IMAGE_JOB,
            # 0: FirestoreFunctions.TRAINING_JOB,
            1: FirestoreFunctions.REPAINT_IMAGE_JOB,
        }

        while True:
            # check for switching pipelines here
            maybe_switch_pipelines(pipeline=pipeline, pipeline_switch_server=pipeline_switch_server)

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
                    job_to_lock = firestoreFunctions.get_jobs(job_type=job_type, resolutions=[], repaint_mode=repaint_mode, repaint_hyper_mode=use_hyper)
                    if job_to_lock is not None:
                        print(f'\nlocking job({job_to_lock.id}) ({job_type})...')
                        firestoreFunctions.lock_job(job_type=job_type, job=job_to_lock)
                        print(f'job locked! keeping same job type ({job_type}) for next loop, sleeping 1 second...')
                        job_order -= 1
                        time.sleep(0.25)
                        firestoreFunctions.db.collection("repainterServers").document(machine_id).set(
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

                        firestoreFunctions.db.collection("repainterServers").document(machine_id).set(
                            {
                                "lastActiveTime": datetime.datetime.now(datetime.timezone.utc),
                                "isBusy": False
                            }
                        )

                        time.sleep(5)
                        continue
            FirestoreFunctions.job_id = started_job.id
            # download the file to the appropriate folder
            image_path, mask_path = job_processor.download_image(started_job)
            # set the downloaded image path
            job_dict = started_job.to_dict()
            job_dict["id"] = started_job.id
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
            # load pipeline here to test imgage processor
            # if pipeline is None:
            #     pipeline = get_pipeline(r, is_server)

            if int(job_dict["mode"]) == 0 and pipeline.model != "t2i":
                firestoreFunctions.db.collection("repainterServers").document(machine_id).set(
                    {
                        "lastActiveTime": datetime.datetime.now(datetime.timezone.utc),
                        "isBusy": True
                    }
                )
                print(f"pipeline model ({pipeline.model}) and job mode {job_dict['mode']} does not match!")
                print("switching to t2i")
                # pipeline.switch_pipeline("t2i")

            elif int(job_dict["mode"]) == 1 and pipeline.model != "fill":
                firestoreFunctions.db.collection("repainterServers").document(machine_id).set(
                    {
                        "lastActiveTime": datetime.datetime.now(datetime.timezone.utc),
                        "isBusy": True
                    }
                )
                print(f"pipeline model ({pipeline.model}) and job mode {job_dict['mode']} does not match!")
                print("switching to fill")
                # pipeline.switch_pipeline("fill")
            else:
                firestoreFunctions.db.collection("repainterServers").document(machine_id).set(
                    {
                        "lastActiveTime": datetime.datetime.now(datetime.timezone.utc),
                        "isBusy": True
                    }
                )

            result = run_redresser_flux_process(
                pipeline=pipeline, options=options,
                img_client=img_client, pipe_server=pipe_server, job_processor=job_processor)

            if not result:
                print("job failed?")
            else:
                # maybe_create_faceswap_job(job_dict)

                job_processor.complete_job(job_dict)
                print("job completed")

            firestoreFunctions.db.collection("repainterServers").document(machine_id).set(
                {
                    "lastActiveTime": datetime.datetime.now(datetime.timezone.utc),
                    "isBusy": True
                }
            )
            job_order -= 1
    else:
        if r == "sd15-v2v":
            settings = CogSettings()
        if r == "cog-i2v":
            settings = CogSettings()
        elif r == "wan-480":
            settings = WanSettings()
        else:
            settings = RedresserSettings()
        while True:
            settings.set_options()
            # check for switching pipelines here after user input for local
            maybe_switch_pipelines(pipeline=pipeline, pipeline_switch_server=pipeline_switch_server)

            runs = int(settings.options.get("runs", 1))
            for _ in range(runs):
                # load pipeline here to test imgage processor
                # if pipeline is None:
                #     pipeline = get_pipeline(r, is_server)

                result = run_redresser_flux_process(
                    pipeline=pipeline, options=settings.options,
                    img_client=img_client, pipe_server=pipe_server, job_processor=None)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--r", default='flux-fill')
    parser.add_argument("-s", "--is_server", default=True, action='store_true')
    parser.add_argument("-m", "--machine_id", default='OVERLORD4-0')
    parser.add_argument("-t", "--use_hyper", default=False, action='store_true')
    parser.add_argument("-p", "--socket_port", default="5000")

    args = parser.parse_args()
    r = args.r
    is_server = args.is_server
    machine_id = args.machine_id
    use_hyper = args.use_hyper
    socket_port = int(args.socket_port)
    pipe_ids = ("flux", "flux-fill", "sd15", "sd15-fill", "cog-i2v", "wan-480", "sd15-v2v")
    if r not in pipe_ids:
        raise ValueError("r must be in one of pipe_ids:", pipe_ids)
    print(f"running with {r},is_server={is_server},machine_id={machine_id}")

    # run("cog-i2v", is_server=False)
    run(r, is_server=is_server, machine_id=machine_id, use_hyper=use_hyper, socket_port=socket_port)
