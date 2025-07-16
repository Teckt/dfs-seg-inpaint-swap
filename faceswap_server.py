import datetime
import os
import PIL.Image as Image
import numpy as np

from iae_firestore_functions import FirestoreFunctions
from safetensors import torch

from dfs_seg_inpaint_swap.faceswap_model import FaceSwapModel

machine_id = "OVERLORD4-0"
FirestoreFunctions.machine_id = machine_id

import shutil
import time


class RepaintFaceswapJobProcesser:

    def __init__(self):
        self.video_swapper = VideoSwapper(
            batch_size=4,
            max_frame_size=4000,
            processes_count=2,
            use_og_extractor=False,
            display_stream=False, use_yolo8=True,
        )



    @staticmethod
    def make_dirs(job_dir=os.path.join('C:' + os.sep, 'deepfakes', 'df-maker-files')):
        RepaintFaceswapJobProcesser.JOB_DIR = job_dir
        if not os.path.exists(RepaintFaceswapJobProcesser.JOB_DIR):
            os.mkdir(RepaintFaceswapJobProcesser.JOB_DIR)

        RepaintFaceswapJobProcesser.USERS_DIR = os.path.join(RepaintFaceswapJobProcesser.JOB_DIR, "users")
        if not os.path.exists(RepaintFaceswapJobProcesser.USERS_DIR):
            os.mkdir(RepaintFaceswapJobProcesser.USERS_DIR)

        RepaintFaceswapJobProcesser.PUBLIC_MODELS_DIR = os.path.join(RepaintFaceswapJobProcesser.JOB_DIR, "models", "dfs")
        RepaintFaceswapJobProcesser.PRETRAINED_MODELS_DIR = os.path.join(RepaintFaceswapJobProcesser.PUBLIC_MODELS_DIR, "pretrained")
        RepaintFaceswapJobProcesser.BASE_MODELS_DIR = os.path.join(RepaintFaceswapJobProcesser.PUBLIC_MODELS_DIR, "base")

    CHECKPOINT_FILE_NAME = "dfsModel.ckpt"

    def download_image(self, job_doc):
        '''

        :param job_doc:
        :return: str: path to downloaded image
        '''
        self.job_doc = job_doc
        self.job_id = job_doc.id
        self.userId = self.job_doc.get("userId")

        # set up job paths
        # make the dir for current user id
        self.userDir = os.path.join(RepaintFaceswapJobProcesser.USERS_DIR, self.userId)
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

        # download the generated/repainted result
        self.imageFileName = job_doc.get("imageFileName")
        firebase_storage_path = self.userRepaintStorageRef + "/" + f"{self.imageFileName}_repainted.png"
        self.input_file_path = self.userRepaintDir + "/" + f"{self.imageFileName}_repainted.png"
        self.output_file_path = self.userRepaintDir + "/" + f"{self.imageFileName}_swapped.png"

        if not FirestoreFunctions.chunked_download(
                firebase_storage_path=firebase_storage_path,
                download_destination_path=self.input_file_path
        ):
            return False

    def download_model(self, job_doc):
        '''
        :param job_doc:
        :return: bool
        '''
        self.job_doc = job_doc
        self.job_id = job_doc.id
        self.userId = self.job_doc.get("userId")
        self.faceModelId = self.job_doc.get("faceModelId")

        # get the face mode doc
        self.model_doc_ref = FirestoreFunctions.usersRef.document(self.userId).collection("faceModels").document(
            self.faceModelId)
        model_doc = self.model_doc_ref.get()
        if not model_doc.exists:
            return False

        try:
            self.modelResolution = model_doc.get("resolution")
            self.modelIterations = int(model_doc.get("iterations"))
            print(f"modelResolution={self.modelResolution}, modelIterations=" + str(self.modelIterations))
        except KeyError:
            print("data missing from faceModel document")
            return False
        except TypeError:
            print(self.userId, self.job_id, self.faceModelId, model_doc.to_dict())
            return False

        # get the model id; if blank (pre-xM version), exit
        self.model_id = model_doc.get("modelId")
        if self.model_id == "":
            return False

        # make sure keys are loaded and valid (should be a set of constants in testing)
        print("model_id", self.model_id)
        assert self.model_id == MODEL_XL

        self.modelResolution = (self.modelResolution, self.modelResolution)

        # make the dir for current user id model root
        self.userModelRootDir = os.path.join(self.userDir, "faceModels")
        if not os.path.exists(self.userModelRootDir):
            os.mkdir(self.userModelRootDir)

        # make the dir for the current user's face model id
        self.userModelDir = os.path.join(self.userModelRootDir, self.faceModelId)
        if not os.path.exists(self.userModelDir):
            os.mkdir(self.userModelDir)

        # set the storage ref for checkpoints, face set zips and all other job specific files
        self.userModelStorageRef = f"users/{self.userId}/faceModels/{self.faceModelId}"

        self.userModelCheckpointFile = os.path.join(self.userModelDir, RepaintFaceswapJobProcesser.CHECKPOINT_FILE_NAME)

        # self.dfs_model = self.models[self.model_id]
        # print(f"loaded model, id: {self.model_id}, resolution: {self.modelResolution}")

        # download the model
        if FirestoreFunctions.exists_in_storage(
                firebase_storage_path=self.userModelStorageRef + "/" + RepaintFaceswapJobProcesser.CHECKPOINT_FILE_NAME
        ):
            FirestoreFunctions.chunked_download(
                firebase_storage_path=self.userModelStorageRef + "/" + RepaintFaceswapJobProcesser.CHECKPOINT_FILE_NAME,
                download_destination_path=self.userModelCheckpointFile
            )
        elif not os.path.exists(self.userModelCheckpointFile):
            return False

    def swap_image(self):
        try:
            pil_img = Image.open(self.input_file_path)
            # pil_img = load_img(self.input_file_path)
            np.array(pil_img).astype(dtype='uint8')
        except:
            return False

        self.video_swapper.swap_image(
            input_file=self.input_file_path,
            output_file=self.output_file_path,
            checkpoint_dir=self.userModelDir,
            model_size={ModelID.dfs_224_xM: "xM",
                        ModelID.dfs_224_xS: "xS",
                        ModelID.df_128: "df_128",
                        ModelID.df_256: "df_256",
                        }[self.model_id],
            is_model_dir=True,  # True to NOT use model manager
            max_frame_size=4096,
            use_watermark=False,
        )

    def complete_job(self):
        '''
        uploads image to redresserImageJobs
        :return:
        '''
        secs = time.time()

        if not os.path.exists(self.output_file_path):
            shutil.copyfile(self.input_file_path, self.output_file_path)
        if not FirestoreFunctions.send_job_file(
                local_file_full_path=self.output_file_path,
                file_name=os.path.basename(self.output_file_path),
                firebase_storage_path=self.userRepaintStorageRef,
                required=True
        ):
            return False

        print(f"sending images took {(time.time() - secs):.2f} secs")
        while True:
            try:
                FirestoreFunctions.repaintImageFaceswapJobsRef.document(self.job_id).set({
                    'jobStatus': "ended",
                    'endedTime': int(time.time()),
                }, merge=True)
                break
            except Exception as e:
                print(e, "Failed to update swapping doc jobStatus. retrying in 5 seconds...")
                time.sleep(5)

        return True

    def fail_job(self):

        while True:
            try:
                FirestoreFunctions.repaintImageFaceswapJobsRef.document(self.job_id).set({
                    'jobStatus': "failed",
                    'endedTime': int(time.time()),
                }, merge=True)
                break
            except Exception as e:
                print(e, "Failed to update swapping doc jobStatus. retrying in 5 seconds...")
                time.sleep(5)

        return True


if __name__ == "__main__":

    job_processor = RepaintFaceswapJobProcesser()
    job_processor.make_dirs()

    firestoreFunctions = FirestoreFunctions()
    job_order = 0
    job_orders = {
        0: FirestoreFunctions.REPAINT_IMAGE_FACESWAP_JOB,
        # 0: FirestoreFunctions.TRAINING_JOB,
        1: FirestoreFunctions.REPAINT_IMAGE_FACESWAP_JOB,
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
                time.sleep(0.25)
                continue
            else:
                job_to_lock = firestoreFunctions.get_jobs_compat(job_type=job_type)
                if job_to_lock is not None:
                    print(f'\nlocking job({job_to_lock.id}) ({job_type})...')
                    firestoreFunctions.lock_job(job_type=job_type, job=job_to_lock)
                    print(f'job locked! keeping same job type ({job_type}) for next loop, sleeping 1 second...')
                    job_order -= 1
                    time.sleep(0.25)
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

        # set the downloaded image path
        job_dict = started_job.to_dict()
        job_dict["id"] = started_job.id
        options = job_dict.copy()
        print("options", options)

        if not job_processor.download_image(started_job):
            job_processor.fail_job()
        if not job_processor.download_model(started_job):
            job_processor.fail_job()
        if not job_processor.swap_image():
            job_processor.fail_job()
        job_processor.complete_job()

        job_order -= 1


if __name__ == "__main__":
    image_size = 320
    latent_dim = 512
    checkpoint = f"faceswap/autoencoder_{image_size}_{latent_dim}.pth"

    face_swapper = RepaintFaceswapJobProcesser(image_size=image_size, latent_dim=latent_dim, checkpoint=checkpoint)