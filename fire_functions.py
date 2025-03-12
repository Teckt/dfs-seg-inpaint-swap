import datetime
import random
import shutil
import threading
import time
from datetime import datetime, timedelta
import json

import cv2
import google
import requests
import os
import ssl

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
from firebase_admin import messaging
# from firebase_admin import auth
# from idna import unicode, re
import google.api_core.exceptions as google_exceptions
from google.cloud.firestore_v1.base_query import FieldFilter
from skimage import io
from CONSTANTS import *

class UserMeta:
    def __init__(self, user_id, sign_up_version, purchases, current_version, credits):
        self.user_id = user_id
        self.sign_up_version = sign_up_version
        self.purchases = purchases
        self.current_version = current_version
        self.credits = credits


class FirestoreFunctions:
    cred = credentials.Certificate(os.path.join("..", CRED_PATH))
    try:
        app = firebase_admin.initialize_app(cred, {'storageBucket': STORAGE_BUCKET_ID})
    except ValueError:
        app = firebase_admin.get_app()

    db = firestore.client()
    bucket = storage.bucket()

    firebase_iae_model_dir_name = "faceModels"
    firebase_face_set_root_dir_name = "iaeFaceSets"
    firebase_face_set_zipped_file_name = "out.zip"

    trainingJobsRef = db.collection(u'trainingJobs')
    swappingJobsRef = db.collection(u'swappingJobs')
    swapImageJobsRef = db.collection(u'swapImageJobs')
    repaintImageJobsRef = db.collection(u'redresserImageJobs')
    usersRef = db.collection(u'users')

    currentJobRef = None
    currentUserRef = None
    currentUserFaceModelRef = None
    currentUserJobIaeModelStorageRef = None

    currentUserFaceSetTargetDir = None
    currentUserFaceSetSourceDir = None
    currentUserModelDir = None

    usersDir = None

    pretrainedModelDir = None
    baseModelDir = None

    samples_target_Xtrain = None
    samples_target_ytarget = None
    samples_target = None

    jobIterations = None
    faceModelIterations = 0
    machine_id = ""

    SWAP_IMAGE_JOB = "swapImageJob"
    SWAPPING_JOB = "swappingJob"
    TRAINING_JOB = "trainingJob"

    REPAINT_IMAGE_JOB = 'repaintImageJob'

    def __init__(self):
        job_dir = JOB_DIR
        FirestoreFunctions.globalJobDir = job_dir
        FirestoreFunctions.usersDir = os.path.join(job_dir, "users")
        FirestoreFunctions.pretrainedModelDir = os.path.join(job_dir, "models", "dfs", "pretrained")
        FirestoreFunctions.baseModelDir = os.path.join(job_dir, "models", "dfs")

        self.jobId = None
        self.userId = None
        self.modelId = None

        self.modelResolution = None
        self.modelBase = None
        self.modelPretrainedTier = None

    @staticmethod
    def human_time(timestamp):
        timestamp = int(str(timestamp)[:10])

        # Set the target timezone
        target_timezone_offset = -5  # Replace with your desired timezone offset in hours

        # Calculate the target timezone offset in seconds
        target_timezone_offset_seconds = target_timezone_offset * 3600

        # Calculate the current time in the target timezone
        current_time_target_timezone = datetime.utcfromtimestamp(timestamp) + timedelta(
            seconds=target_timezone_offset_seconds)

        # Format the time as a human-readable string
        formatted_time_24 = current_time_target_timezone.strftime('%Y-%m-%d %H:%M:%S')

        # Format the time as a human-readable string with AM/PM
        formatted_time_12 = current_time_target_timezone.strftime('%Y-%m-%d %I:%M:%S %p')

        # Display the result
        # print(f"Current time in target timezone: {formatted_time_12}, {formatted_time_24}")
        return formatted_time_12, formatted_time_24

    @staticmethod
    def get_user_meta(user_id):
        purchases = FirestoreFunctions.db.collection(u'users').document(user_id).collection(u'purchases').get()
        user_doc = FirestoreFunctions.db.collection(u'users').document(user_id).get()
        try:
            sign_up_version = int(user_doc.get("signUpVersionCode"))
        except KeyError:
            sign_up_version = 0

        try:
            current_version = int(user_doc.get("currentVersionCode"))
        except KeyError:
            current_version = 0

        try:
            credits = int(user_doc.get("credits"))
        except KeyError:
            credits = 0

        user_meta = UserMeta(
            user_id=user_id,
            sign_up_version=sign_up_version,
            current_version=current_version,
            purchases=purchases,
            credits=credits,
        )

        return user_meta

    @staticmethod
    def get_user_revenue(user_id=None, purchases=None):
        if purchases is None and user_id is None:
            raise ValueError("user_id and purchases cannot both be None")

        total_revenue = 0
        docs = FirestoreFunctions.db.collection(u'users').document(user_id).collection(u'purchases').get() if purchases is None else purchases
        for doc in docs:

            if doc.get('processed') is False:
                continue

            # errorMsg = doc.get("errorMsg")
            purchaseTime = doc.get("purchaseTime")
            sku = doc.get("sku")
            purchaseToken = doc.get("purchaseToken")
            processed = doc.get("processed")

            if sku == 'dfstudio_credits_100':
                total_revenue += 1.99
            elif sku == 'dfstudio_credits_300':
                total_revenue += 5.49
            elif sku == 'dfstudio_credits_1000':
                total_revenue += 14.99
        return total_revenue
        print(f'total_purchases: {len(docs)}, total_revenue: ${round(total_revenue, 2)}')


    @staticmethod
    def reply_user_issue(issue_id, message):
        city = {
            "content": message,
            "published": int(time.time()),
            "seen": False,
            "userId": "",
        }
        user_issue_messages_ref = FirestoreFunctions.db.collection(u'userIssues').document(issue_id).collection(u'messages')
        update_time, message_ref = user_issue_messages_ref.add(city)
        print(f"Added document with id {message_ref.id} at {update_time}")
        pass

    @staticmethod
    def send_notification(user_id, message):
        fcmTokens = []
        userDoc = FirestoreFunctions.usersRef.document(user_id).get()
        # get the fck token if exists, otherwise skip user
        try:
            fcmToken = userDoc.get('fcmToken')
            fcmTokens.append(fcmToken)
        except KeyError:
            print("no token found!")
            return False

        # set the notification details
        notification = messaging.Notification(
            title=message,
        )
        # prepare the multicast object to send
        message = messaging.MulticastMessage(tokens=fcmTokens, notification=notification)
        # send the messages
        try:
            response = messaging.send_multicast(message)
        except:
            print("messaging error!")
            return False


        failed_sends = 0
        if response.failure_count > 0:
            responses = response.responses
            failed_tokens = []
            for idx, resp in enumerate(responses):
                if not resp.success:
                    failed_sends += 1
                    print(f'fcmTokens[idx]: {fcmTokens[idx]}, resp: {resp.exception},')
                    # The order of responses corresponds to the order of the registration tokens.
                    failed_tokens.append(fcmTokens[idx])
            print(f'{failed_sends} tokens that caused failures : {failed_tokens}')
            return False
        else:
            print(f'Notification sent to {user_id}')
            return True

    @staticmethod
    def send_job_complete_notification(job_type, user_id, message='Job complete'):
        fcmTokens = []
        userDoc = FirestoreFunctions.usersRef.document(user_id).get()
        # get the fck token if exists, otherwise skip user
        try:
            fcmToken = userDoc.get('fcmToken')
            fcmTokens.append(fcmToken)
        except KeyError:
            return

        # set the notification details
        notification = messaging.Notification(
            title=message,
        )
        # prepare the multicast object to send
        try:
            message = messaging.MulticastMessage(tokens=fcmTokens, notification=notification)
        except ValueError:
            return
        # send the messages

        try:
            response = messaging.send_multicast(message)
        except:
            print('error nptie')
            return

        failed_sends = 0
        if response.failure_count > 0:
            responses = response.responses
            failed_tokens = []
            for idx, resp in enumerate(responses):
                if not resp.success:
                    failed_sends += 1
                    print(f'fcmTokens[idx]: {fcmTokens[idx]}, resp: {resp.exception},')
                    # The order of responses corresponds to the order of the registration tokens.
                    failed_tokens.append(fcmTokens[idx])
            print(f'{failed_sends} tokens that caused failures : {failed_tokens}')
        else:
            print(f'Job complete notification sent to {user_id}')

    @staticmethod
    def get_user_nan_models(user_id):
        faceModels = FirestoreFunctions.usersRef.document(user_id).collection("faceModels").get()

        nan_faceModels = []

        for faceModel in faceModels:
            faceModelId = faceModel.id
            # check for loss key
            try:
                loss = faceModel.get("decoder_a_loss")
                iterations = faceModel.get("iterations")
                resolution =  faceModel.get("resolution")
            except KeyError:
                continue

            if loss == "NaN":
                nan_faceModels.append({
                    "faceModelId":faceModelId,
                    "iterations":iterations,
                    "resolution":resolution,
                })

        return nan_faceModels

    def get_job_ref(self, job_type):
        if job_type == FirestoreFunctions.TRAINING_JOB:
            ref = FirestoreFunctions.trainingJobsRef
        elif job_type == FirestoreFunctions.SWAPPING_JOB:
            ref = FirestoreFunctions.swappingJobsRef
        elif job_type == FirestoreFunctions.SWAP_IMAGE_JOB:
            ref = FirestoreFunctions.swapImageJobsRef
        elif job_type == FirestoreFunctions.REPAINT_IMAGE_JOB:
            ref = FirestoreFunctions.repaintImageJobsRef
        else:
            ref = None

        return ref
    def get_started_jobs(self, job_type):
        # returns the first started job
        ref = self.get_job_ref(job_type)
        if ref is None:
            return None

        try:
            jobs = ref.where(filter=FieldFilter('machine_id', '==', FirestoreFunctions.machine_id)).where(filter=FieldFilter('jobStatus', '==', 'started')).get()
        except:
            print("google.api_core.exceptions.Unknown, sleeping 5 seconds...")
            time.sleep(5)
            return None

        started_job = None
        for job in jobs:
            started_job = job

        if started_job is not None:
            # clear the lock for any remaining queued jobs for this machine id
            queued_jobs = ref.where(filter=FieldFilter('machine_id', '==', FirestoreFunctions.machine_id)).where(filter=FieldFilter('jobStatus', '==', 'queued')).get()
            for job in queued_jobs:
                ref.document(job.id).set(
                    {
                        'machine_id': "",
                    },
                    merge=True
                )

        return started_job

    def get_locked_jobs(self, job_type):
        # gets oldest job with this machine_id and locked
        ref = self.get_job_ref(job_type)
        if ref is None:
            return None

        try:
            queued_jobs = ref.where(filter=FieldFilter('machine_id', '==', FirestoreFunctions.machine_id)).where(
                filter=FieldFilter('jobStatus', '==', 'queued')).get()
        except:
            print("google.api_core.exceptions.Unknown, sleeping 5 seconds...")
            time.sleep(5)
            return None

        job_to_start = None
        for job in queued_jobs:
            if job_to_start is None or job_to_start.get("queuedTime") < job.get("queuedTime"):
                job_to_start = job

        return job_to_start

    def start_job(self, job_type, job):
        # sets job to start
        ref = self.get_job_ref(job_type)
        if ref is None:
            return
        while True:
            try:
                ref.document(job.id).set(
                    {
                        'jobStatus': 'started',
                        'startedTime': int(time.time()),
                    },
                    merge=True
                )
                break
            except:
                print("error, trying again in 5")
                time.sleep(5)
        if job_type == FirestoreFunctions.SWAPPING_JOB:
            # change status of swapVideoJob too if exists
            try:
                swapVideoJobId = job.get("swapVideoJobId")
            except KeyError:
                swapVideoJobId = ""
            if swapVideoJobId != "":
                user_id = job.get("userId")
                swapVideoJobRef = FirestoreFunctions.usersRef.document(user_id).collection("swapVideoJobs").document(swapVideoJobId)
                swapVideoJob = swapVideoJobRef.get()
                if swapVideoJob.exists:
                    while True:
                        try:
                            swapVideoJobRef.set(
                                {
                                    'jobStatus': 'started',
                                    'startedTime': int(time.time()),
                                },
                                merge=True
                            )
                            break
                        except:
                            print("error, trying again in 5")
                            time.sleep(5)


        if job_type == FirestoreFunctions.SWAP_IMAGE_JOB:
            # change status of swapVideoJob too if exists
            try:
                swapImageJobId = job.get("swapImageJobId")
            except KeyError:
                swapImageJobId = ""
            if swapImageJobId != "":
                user_id = job.get("userId")
                swapImageJobRef = FirestoreFunctions.usersRef.document(user_id).collection("swapImageJobs").document(swapImageJobId)
                while True:
                    try:
                        swapImageJob = swapImageJobRef.get()
                        if swapImageJob.exists:
                            swapImageJobRef.set(
                                {
                                    'jobStatus': 'started',
                                    'startedTime': int(time.time()),
                                },
                                merge=True
                            )
                        break
                    except:
                        print("error, trying again in 5")
                        time.sleep(5)

    def get_jobs(self, job_type, resolutions=(128, 224, 256)):
        ref = self.get_job_ref(job_type)
        if ref is None:
            return None

        if 128 in resolutions and 224 in resolutions and 256 in resolutions:
            jobs = ref.where(filter=FieldFilter("jobStatus", "==", "queued")).order_by(u'queuedTime', direction=firestore.firestore.Query.ASCENDING).get()
        else:
            if job_type == FirestoreFunctions.REPAINT_IMAGE_JOB:
                jobs = ref.where(filter=FieldFilter("jobStatus", "==", "queued"))\
                    .order_by(u'queuedTime', direction=firestore.firestore.Query.ASCENDING).get()
            else:
                jobs = ref.where(filter=FieldFilter("jobStatus", "==", "queued")).where(filter=FieldFilter("resolution", "in", resolutions)).order_by(u'queuedTime', direction=firestore.firestore.Query.ASCENDING).get()

        for job in jobs:
            user_id = str(job.get("userId"))
            print(f"user_id:{user_id}, job_id={job.id}")
            # if user_id in [
            #     "IgN7nUZ84fMkvrVPK1A2VjzE5Ge2",
            #
            # ]:
            #     continue

            try:
                machine_id = str(job.get("machine_id"))
            except KeyError:
                machine_id = ""

            if machine_id == "":
                return job

        return None

    def lock_job(self, job_type, job):
        ref = self.get_job_ref(job_type)
        if ref is None:
            return
#google.api_core.exceptions.InternalServerError: 500 An internal error occurred.
        ref.document(job.id).set(
            {
                'machine_id': FirestoreFunctions.machine_id,
            },
            merge=True
        )

    def get_swappingJob_files(self, jobId):

        self.jobId = jobId

        job = FirestoreFunctions.swappingJobsRef.document(jobId).get()
        try:
            self.userId = job.get("userId")
            self.modelId = job.get("faceModelId")
            faceModelIsPretrain = job.get("faceModelIsPretrain")
        except KeyError:
            print("data missing from swappingJob document")
            return False, False, False, False

        FirestoreFunctions.currentUserJobIaeModelStorageRef = f"users/{self.userId}/faceModels/{self.modelId}"

        modelDoc = FirestoreFunctions.usersRef.document(self.userId).collection("faceModels").document(self.modelId).get()
        try:

            self.modelBase = modelDoc.get("base")
            self.modelPretrainedTier = modelDoc.get("pretrainedTier")
            if faceModelIsPretrain:
                FirestoreFunctions.faceModelIterations = 0
                self.modelResolution = (224, 224)
            else:
                self.modelResolution = modelDoc.get("resolution")
                self.modelResolution = (self.modelResolution, self.modelResolution)
                FirestoreFunctions.faceModelIterations = int(modelDoc.get("iterations"))
            print("FirestoreFunctions.faceModelIterations=" + str(FirestoreFunctions.faceModelIterations))
        except KeyError:
            print("data missing from faceModel document")
            return False, False, False, False

        FirestoreFunctions.currentUserRef = FirestoreFunctions.db.collection(u'users')

        # face set files
        user_face_set_root_dir = os.path.join(FirestoreFunctions.usersDir, self.userId,
                                              FirestoreFunctions.firebase_face_set_root_dir_name)
        if not os.path.exists(user_face_set_root_dir):
            os.mkdir(user_face_set_root_dir)

        # get the face set for the face model. Should've uploaded the face set to server due to training
        # local
        if faceModelIsPretrain:
            pretrainedModelDir = os.path.join(FirestoreFunctions.pretrainedModelDir, self.modelId)
            if not os.path.exists(pretrainedModelDir):
                raise FileNotFoundError(f"pretrainedModelDir({pretrainedModelDir}) does not exist")
                # os.mkdir(pretrainedModelDir)
            FirestoreFunctions.currentUserFaceSetTargetDir = os.path.join(pretrainedModelDir, "faceSet")
            if not os.path.exists(FirestoreFunctions.currentUserFaceSetTargetDir):
                raise FileNotFoundError(f"currentUserFaceSetTargetDir ({FirestoreFunctions.currentUserFaceSetTargetDir}) does not exist")
                # os.mkdir(FirestoreFunctions.currentUserFaceSetTargetDir)
            # server
            FirestoreFunctions.currentUserFaceSetTargetRef = f'faceModels/{self.modelId}/out.zip'
        else:
            FirestoreFunctions.currentUserFaceSetTargetDir = os.path.join(user_face_set_root_dir, self.modelId)
            # server
            FirestoreFunctions.currentUserFaceSetTargetRef = f'users/{self.userId}/{FirestoreFunctions.firebase_face_set_root_dir_name}/{self.modelId}/out.zip'

        # local
        FirestoreFunctions.currentUserFaceSetSourceDir = os.path.join(user_face_set_root_dir, "swappingJob_extracted_faces")
        # server
        FirestoreFunctions.currentUserFaceSetSourceRef = f'users/{self.userId}/faceModels/{self.modelId}/extracted_faces.zip'

        print(f'FirestoreFunctions.currentUserFaceSetTargetRef={FirestoreFunctions.currentUserFaceSetTargetRef}')
        print(f'FirestoreFunctions.currentUserFaceSetTargetDir={FirestoreFunctions.currentUserFaceSetTargetDir}')
        print(f'FirestoreFunctions.currentUserFaceSetSourceRef={FirestoreFunctions.currentUserFaceSetSourceRef}')
        print(f'FirestoreFunctions.currentUserFaceSetSourceDir={FirestoreFunctions.currentUserFaceSetSourceDir}')

        # if not faceModelIsPretrain:
        if not self.get_user_face_set(user_face_set_ref=FirestoreFunctions.currentUserFaceSetTargetRef,
                                      user_face_set_dir=FirestoreFunctions.currentUserFaceSetTargetDir):
            return False, False, False, False
        if not self.get_user_face_set(user_face_set_ref=FirestoreFunctions.currentUserFaceSetSourceRef,
                                      user_face_set_dir=FirestoreFunctions.currentUserFaceSetSourceDir):
            return False, False, False, False

        user_model_root_dir = os.path.join(FirestoreFunctions.usersDir, self.userId,
                                           FirestoreFunctions.firebase_iae_model_dir_name)
        if not os.path.exists(user_model_root_dir):
            os.mkdir(user_model_root_dir)
        # else:
        #     shutil.rmtree(user_model_dir)
        #     os.mkdir(user_model_dir)
        if faceModelIsPretrain:
            FirestoreFunctions.currentUserModelDir = pretrainedModelDir
        else:
            FirestoreFunctions.currentUserModelDir = os.path.join(user_model_root_dir, self.modelId)
            if not os.path.exists(FirestoreFunctions.currentUserModelDir):
                os.mkdir(FirestoreFunctions.currentUserModelDir)

        swapping_job_swapped_faces_path = os.path.join(user_face_set_root_dir, "swappingJob_swapped_faces")
        if not os.path.exists(swapping_job_swapped_faces_path):
            os.mkdir(swapping_job_swapped_faces_path)
        return FirestoreFunctions.currentUserFaceSetTargetDir, FirestoreFunctions.currentUserFaceSetSourceDir, swapping_job_swapped_faces_path, FirestoreFunctions.currentUserModelDir

    def get_trainingJob_files(self, jobId):
        self.jobId = jobId

        job = FirestoreFunctions.trainingJobsRef.document(jobId).get()
        try:
            self.userId = job.get("userId")
            self.modelId = job.get("modelId")
            # faceSetTargetIndex = job.get('faceSetTargetIndex')
            # faceSetSourceIndex = job.get('faceSetSourceIndex')

            faceSetTargetId = job.get('faceSetTargetId')
            faceSetSourceId = job.get('faceSetSourceId')

            # jobStatus = job.get("jobStatus")
            # queuedTime = int(job.get('queuedTime'))

            starting_iterations = int(job.get('starting_iterations'))
            FirestoreFunctions.jobIterations = int(job.get("jobIterations"))

        except KeyError:
            print("data missing from trainingJob document")
            return False, False, False, False, False

        modelDoc = FirestoreFunctions.usersRef.document(self.userId).collection("faceModels").document(self.modelId).get()
        try:
            self.modelResolution = modelDoc.get("resolution")
            self.modelResolution = (self.modelResolution, self.modelResolution)
            self.modelBase = modelDoc.get("base")
            self.modelPretrainedTier = modelDoc.get("pretrainedTier")

            FirestoreFunctions.faceModelIterations = int(modelDoc.get("iterations"))
            print("FirestoreFunctions.faceModelIterations=" + str(FirestoreFunctions.faceModelIterations))
        except KeyError:
            print("data missing from faceModel document")
            return False, False, False, False

        FirestoreFunctions.currentUserFaceModelRef = FirestoreFunctions.db.collection(u'users').document(
            f"{self.userId}/faceModels/{self.modelId}")
        FirestoreFunctions.currentUserJobIaeModelStorageRef = f"users/{self.userId}/faceModels/{self.modelId}"

        return self.get_job_files(jobId, self.userId, faceSetTargetId, faceSetSourceId, self.modelId, starting_iterations)

    def get_job_files(self, jobId, userId, faceSetTargetId, faceSetSourceId, modelId, starting_iterations):
        FirestoreFunctions.currentJobRef = FirestoreFunctions.trainingJobsRef.document(jobId)
        FirestoreFunctions.currentUserRef = FirestoreFunctions.db.collection(u'users')

        # face set files
        user_face_set_root_dir = os.path.join(FirestoreFunctions.usersDir, userId, FirestoreFunctions.firebase_face_set_root_dir_name)
        if not os.path.exists(user_face_set_root_dir):
            os.mkdir(user_face_set_root_dir)

        FirestoreFunctions.currentUserFaceSetTargetDir = os.path.join(user_face_set_root_dir, faceSetTargetId)
        FirestoreFunctions.currentUserFaceSetSourceDir = os.path.join(user_face_set_root_dir, faceSetSourceId)

        FirestoreFunctions.currentUserFaceSetTargetRef = f'users/{userId}/{FirestoreFunctions.firebase_face_set_root_dir_name}/{faceSetTargetId}/out.zip'
        FirestoreFunctions.currentUserFaceSetSourceRef = f'users/{userId}/{FirestoreFunctions.firebase_face_set_root_dir_name}/{faceSetSourceId}/out.zip'

        # if not self.get_user_face_set(user_face_set_ref=FirestoreFunctions.currentUserFaceSetTargetRef,
        #                               user_face_set_dir=FirestoreFunctions.currentUserFaceSetTargetDir):
        #     return False, False, False, False
        # if not self.get_user_face_set(user_face_set_ref=FirestoreFunctions.currentUserFaceSetSourceRef,
        #                               user_face_set_dir=FirestoreFunctions.currentUserFaceSetSourceDir):
        #     return False, False, False, False

        user_model_root_dir = os.path.join(FirestoreFunctions.usersDir, userId, FirestoreFunctions.firebase_iae_model_dir_name)
        if not os.path.exists(user_model_root_dir):
            os.mkdir(user_model_root_dir)
        # else:
        #     shutil.rmtree(user_model_dir)
        #     os.mkdir(user_model_dir)

        FirestoreFunctions.currentUserModelDir = os.path.join(user_model_root_dir, modelId)
        if not os.path.exists(FirestoreFunctions.currentUserModelDir):
            os.mkdir(FirestoreFunctions.currentUserModelDir)

        FirestoreFunctions.currentJobRef.update({
            'iterations': 0,
        })

        return FirestoreFunctions.currentUserFaceSetTargetDir, FirestoreFunctions.currentUserFaceSetSourceDir,\
               FirestoreFunctions.currentUserModelDir, starting_iterations

    def get_user_face_set(self, user_face_set_ref, user_face_set_dir):
        if not os.path.exists(user_face_set_dir):
            os.mkdir(user_face_set_dir)
        # else:
        #     shutil.rmtree(user_face_set_dir)
        #     os.mkdir(user_face_set_dir)

        user_face_set_file = os.path.join(user_face_set_dir, FirestoreFunctions.firebase_face_set_zipped_file_name)
        print(f"user_face_set_file={user_face_set_file}, exists={os.path.exists(user_face_set_file)}")

        if FirestoreFunctions.chunked_download(
            firebase_storage_path=user_face_set_ref,
            download_destination_path=user_face_set_file
        ):
            # delete old face set files but leave the zip
            old_face_set_files = os.listdir(user_face_set_dir)
            for file in old_face_set_files:
                if file == "mask":
                    shutil.rmtree(os.path.join(user_face_set_dir, file))
                if file == "mask_eyes":
                    shutil.rmtree(os.path.join(user_face_set_dir, file))
                if file == "mask_mouth":
                    shutil.rmtree(os.path.join(user_face_set_dir, file))
                elif file.endswith(".jpg") or file.endswith(".png"):
                    os.remove(os.path.join(user_face_set_dir, file))

            shutil.unpack_archive(
                filename=user_face_set_file,
                extract_dir=user_face_set_dir
            )
            return True
        else:
            return False

    def complete_job(self):
        # job = self.trainingJobs_ref.document(self.jobId).get()
        # try:
        #     userId = job.get("userId")
        #     modelId = job.get("modelId")
        #     # modelName = job.get("modelName")
        #     faceSetTargetIndex = job.get('faceSetTargetIndex')
        #     faceSetSourceIndex = job.get('faceSetSourceIndex')
        #     # faceSetSrcName = job.get('faceSetSrcName')
        #     # faceSetDstName = job.get('faceSetDstName')
        #     # resolution = job.get('resolution')
        # except KeyError:
        #     print("data missing from trainingJob document")
        #     return False
        # userModelDoc = FirestoreFunctions.currentUserFaceModelRef.get()
        # try:
        #     previous_iterations = userModelDoc.get("iterations")
        # except KeyError:
        #     previous_iterations = 0
        current_iters = int(FirestoreFunctions.faceModelIterations + FirestoreFunctions.jobIterations)
        while True:
            try:
                FirestoreFunctions.currentUserFaceModelRef.set({
                    'iterations': current_iters,
                }, merge=True)
                FirestoreFunctions.trainingJobsRef.document(self.jobId).update({
                    'jobStatus': "ended",
                    'endedTime': int(time.time()),
                })
                break
            except:
                print("Failed to update training doc jobStatus. retrying in 5 seconds...")






        # firebase_tflite_model_path = f"users/{userId}/iaeModels/{modelId}"
        # user_face_set_root_dir = os.path.join(FirestoreFunctions.usersDir, userId, FirestoreFunctions.firebase_face_set_dir_name)
        # user_face_set_src_dir = os.path.join(user_face_set_root_dir, str(faceSetTargetIndex))
        # user_face_set_dst_dir = os.path.join(user_face_set_root_dir, str(faceSetSourceIndex))
        #
        # user_face_set_src_images = [file for file in os.listdir(user_face_set_src_dir) if file.endswith(".jpg")]
        # user_face_set_dst_images = [file for file in os.listdir(user_face_set_dst_dir) if file.endswith(".jpg")]
        #
        # shutil.copy(os.path.join(user_face_set_src_dir, user_face_set_src_images[0]),
        #             os.path.join(user_face_set_root_dir, 'faceSetSrcSample.jpg'))
        #
        # shutil.copy(os.path.join(user_face_set_dst_dir, user_face_set_dst_images[0]),
        #             os.path.join(user_face_set_root_dir, 'faceSetDstSample.jpg'))

        # FirestoreFunctions.send_job_file(
        #     firebase_storage_path=firebase_tflite_model_path,
        #     local_file_full_path=user_face_set_root_dir,
        #     file_name='faceSetDstSample.jpg',
        #     required=True
        # )
        #
        # FirestoreFunctions.send_job_file(
        #     firebase_storage_path=firebase_tflite_model_path,
        #     local_file_full_path=user_face_set_root_dir,
        #     file_name='faceSetSrcSample.jpg',
        #     required=True
        # )

        return True

    @staticmethod
    def exists_in_storage(firebase_storage_path):
        blob = None
        print("Getting file size...")
        while blob is None:
            try:
                blob = FirestoreFunctions.bucket.get_blob(blob_name=firebase_storage_path)
                if blob is None:
                    return False
            except ConnectionResetError:
                print("\rConnectionResetError", end="")
                time.sleep(5)
            except ConnectionError:
                print("\rConnectionError blob is none, sleeping 5", end="")
                time.sleep(5)
            except requests.exceptions.ConnectionError:
                print("\rrequests.exceptions.ConnectionError blob is none, sleeping 5", end="")
                time.sleep(5)
            except google_exceptions.ServiceUnavailable:
                print("\rgoogle service unavailable, sleeping 5", end="")
                time.sleep(5)
        total_file_size = blob.size
        return True

    @staticmethod
    def download_user_face_set(userId, faceSetId, download_destination_path):
        user_face_set_ref = f'users/{userId}/{FirestoreFunctions.firebase_face_set_root_dir_name}/{faceSetId}/out.zip'
        return FirestoreFunctions.chunked_download(
            firebase_storage_path=user_face_set_ref,
            download_destination_path=download_destination_path
        )

    @staticmethod
    def chunked_download(firebase_storage_path, download_destination_path):
        blob = None
        print(f"Getting file size from {firebase_storage_path}...")
        blob = FirestoreFunctions.bucket.get_blob(blob_name=firebase_storage_path)
        if blob is None:
            print("blob is None")
            return False
        # while blob is None:
        #     try:
        #         blob = FirestoreFunctions.bucket.get_blob(blob_name=firebase_storage_path)
        #         if blob is None:
        #             print("blob is None")
        #             return False
        #     except ConnectionResetError:
        #         print("\rConnectionResetError", end="")
        #         time.sleep(5)
        #     except ConnectionError:
        #         print("\rConnectionError blob is none, sleeping 5", end="")
        #         time.sleep(5)
        #     except requests.exceptions.ConnectionError:
        #         print("\rrequests.exceptions.ConnectionError blob is none, sleeping 5", end="")
        #         time.sleep(5)
        #     except ServiceUnavailable:
        #         print("\rgoogle service unavailable, sleeping 5", end="")
        #         time.sleep(5)
        total_file_size = blob.size
        print(f"Total size: {total_file_size} bytes...")

        file_size = 0
        # get file size of target if exists
        print(f"os.path.exists(download_destination_path[{download_destination_path}]) = {os.path.exists(download_destination_path.strip())}")
        if os.path.exists(download_destination_path):
            file_size = os.path.getsize(download_destination_path)
            print(f"File already exists with size:{file_size}")

        if total_file_size == file_size:
            print("File already fully downloaded")
            return True
        else:
            print(f"\rDownloading {firebase_storage_path} to {download_destination_path}")
        # set the name of the chunked file
        chunk_file = download_destination_path + ".part"
        # delete any existing chunks
        if os.path.isfile(chunk_file):
            print(f"deleting prev chuk file: {chunk_file}")
            os.unlink(chunk_file)
        # start loop
        while file_size < total_file_size:
            # refresh file size
            if os.path.isfile(download_destination_path):
                file_size = os.path.getsize(download_destination_path)
            # determine chunk size based on existing bytes and maximum chunk size
            chunk_size = min(1024 * 1024 * 16, total_file_size - file_size) + file_size
            # attempt to get chunk from cloud storage
            print(f"chunking {chunk_size}, {file_size}/{total_file_size}")
            try:
                blob.download_to_filename(filename=chunk_file, start=file_size, end=chunk_size - 1)
            except ssl.SSLError:
                # try again after 2 seconds
                print("\rGotten SSLError error. Retrying in 2 seconds...", end="")
                time.sleep(2)
            except requests.exceptions.ChunkedEncodingError:
                # try again after 2 seconds
                print("\rrequests.exceptions.ChunkedEncodingError. Retrying in 2 seconds...", end="")
                time.sleep(2)
            except requests.exceptions.SSLError:
                # try again after 2 seconds
                print("\rGotten SSLError error. Retrying in 2 seconds...", end="")
                time.sleep(2)
            except requests.exceptions.ConnectionError:
                # try again after 2 seconds
                print("\rGotten connection error. Retrying in 2 seconds...", end="")
                time.sleep(2)
            except requests.exceptions.ReadTimeout:
                # try again after 2 seconds
                print("\rGotten timeout error. Retrying in 2 seconds...", end="")
                time.sleep(2)
            except:
                print("\rGotten timeout error. Retrying in 2 seconds...", end="")
                time.sleep(2)
            # append to existing target or creates a new file if not exists
            print("downloaded chunk")
            if os.path.isfile(chunk_file):
                with open(download_destination_path + ".part", 'rb') as file_part:  # open chunk for reading
                    with open(download_destination_path, 'ab') as f:  # open target in append bytes mode for writing chunk to
                        while True:
                            buffer = file_part.read(chunk_size)  # read chunk into memory
                            if not buffer:  # break if reached EOL
                                break
                            f.write(buffer)  # write chunk to target
                            file_size = file_size + len(buffer)  # add to the total size to flag the while loop

                print("\rAppended " + str(os.path.getsize(chunk_file)) + " bytes to current bytes " + str(
                    file_size) + ", remaining bytes: " + str(total_file_size - file_size), end="")
                os.unlink(chunk_file)
        print("Download complete!")
        return True

    @staticmethod
    def send_job_file(firebase_storage_path, local_file_full_path, file_name, required=False):

        firebase_storage_path = firebase_storage_path + '/' + file_name
        link = "file:///" + local_file_full_path.replace('\\', '/')

        print('Sending ' + link + ' to: ' + firebase_storage_path)
        link = link.replace("training__", "")
        print("Parent folder:", link.replace(file_name, ""))

        if os.path.isfile(local_file_full_path):
            blob = FirestoreFunctions.bucket.blob(firebase_storage_path)

            while True:
                try:
                    blob.upload_from_filename(filename=local_file_full_path)

                    blob = FirestoreFunctions.bucket.get_blob(blob_name=firebase_storage_path)
                    if blob is not None:
                        # print('Uploaded, checking file size...')
                        file_size = os.path.getsize(local_file_full_path)
                        total_file_size = blob.size
                        if file_size == total_file_size:
                            break
                        else:
                            print(f'File size doesn\'t match! {total_file_size}/{file_size}')
                    else:
                        print('File doesn\'t exist!')
                except ConnectionResetError:
                    print("ConnectionResetError")
                    if required:
                        time.sleep(5)
                    else:
                        return False
                except ConnectionError:
                    print("ConnectionError blob is none, sleeping 5")
                    if required:
                        time.sleep(5)
                    else:
                        return False
                except requests.exceptions.ConnectionError:
                    print("requests.exceptions.ConnectionError blob is none, sleeping 5")
                    if required:
                        time.sleep(5)
                    else:
                        return False
                except google_exceptions.ServiceUnavailable:
                    print("google service unavailable, sleeping 5")
                    if required:
                        time.sleep(5)
                    else:
                        return False
                except google_exceptions.TooManyRequests:
                    print("google service TooManyRequests, sleeping 5")
                    if required:
                        time.sleep(5)
                    else:
                        return False
                except requests.exceptions.ReadTimeout:
                    print("requests.exceptions.ReadTimeout, sleeping 5")
                    if required:
                        time.sleep(5)
                    else:
                        return False
        else:
            print(f"({local_file_full_path}) file not found")
            return False

        return True

    @staticmethod
    def get_user_purchases(time_range_start, time_range_end=time.time(), include_all=True):
        customer_dict = {}
        read_cost = 0
        docs = FirestoreFunctions.db.collection_group(u'purchases').where('purchaseTime', '>=', time_range_start * 1000).where(
            'purchaseTime', '<=', time_range_end * 1000).order_by(u'purchaseTime',
                                                                  direction=firestore.firestore.Query.ASCENDING).stream()
        for doc in docs:
            read_cost += 1

            if doc.get('processed') is False:
                continue

            try:
                userId = doc.get("userId")
            except KeyError:
                continue

            if userId not in customer_dict.keys():
                print(f"processing", userId)
                user_purchases = FirestoreFunctions.db.collection(u'users').document(userId).collection(u'purchases').stream()
                customer_dict[userId] = 0
                for user_purchase in user_purchases:
                    read_cost += 1

                    if user_purchase.get('processed') is True:
                        sku = doc.get("sku")
                        if sku == 'dfstudio_credits_100':
                            customer_dict[userId] += 1.99
                        elif sku == 'dfstudio_credits_300':
                            customer_dict[userId] += 5.49
                        elif sku == 'dfstudio_credits_1000':
                            customer_dict[userId] += 14.99
                print(f"{userId}:{customer_dict[userId]}")

        total_amount = 0
        for userId, amount in customer_dict.items():
            total_amount += amount
            print(f"{userId}: {amount}")

        print(f"{FirestoreFunctions.human_time(time_range_start)} - {FirestoreFunctions.human_time(time_range_end)}")
        print(f"total customers:", len(customer_dict))
        print(f"total amount:", total_amount)
        print("Read cost:", read_cost)

        return customer_dict

    @staticmethod
    def get_user_purchases_recent(time_range_start, time_range_end=time.time()):
        customer_dict = {}
        read_cost = 0
        docs = FirestoreFunctions.db.collection_group(u'purchases').where('purchaseTime', '>=',
                                                                          time_range_start * 1000).where(
            'purchaseTime', '<=', time_range_end * 1000).order_by(u'purchaseTime',
                                                                  direction=firestore.firestore.Query.ASCENDING).stream()
        for doc in docs:
            read_cost += 1

            if doc.get('processed') is False:
                continue

            try:
                userId = doc.get("userId")
            except KeyError:
                continue

            if userId not in customer_dict.keys():
                user_doc = FirestoreFunctions.usersRef.document(userId).get()
                try:
                    sign_up_version = user_doc.get("signUpVersionCode")
                except KeyError:
                    sign_up_version = 0

                customer_dict[userId] = 0
                customer_dict[userId] = {
                    'amount': 0,
                    'sign_up_version': sign_up_version
                }

            if doc.get('processed') is True:
                sku = doc.get("sku")
                if sku == 'dfstudio_credits_100':
                    customer_dict[userId]['amount'] += 1.99
                elif sku == 'dfstudio_credits_300':
                    customer_dict[userId]['amount'] += 5.49
                elif sku == 'dfstudio_credits_1000':
                    customer_dict[userId]['amount'] += 14.99
            else:
                # print(f"{userId}:{customer_dict[userId]} - NOT PROCESSED")
                continue

            # print(f"{userId}:{customer_dict[userId]}")

        total_amount = 0
        for userId, cust_info in customer_dict.items():
            amount = cust_info['amount']
            sign_up_version = cust_info['sign_up_version']
            # if amount > 2:
            print(f"{userId}: {amount}, sign_up_version: {sign_up_version}")
            total_amount += amount


        print(f"{FirestoreFunctions.human_time(time_range_start)} - {FirestoreFunctions.human_time(time_range_end)}")
        print(f"total customers:", len(customer_dict))
        print(f"total amount:", total_amount)
        print("Read cost:", read_cost)

        return customer_dict

    @staticmethod
    def get_user_issues():
        # with open("promo_0.txt", 'r') as file:
        #     promo_dict = json.load(file)
        #
        # user_issue_ids_do_not_delete = [
        #     "Ekbh5zsJLhc6g4gdGZWj0kmBbxh2"
        #     "bTCvP5NBXfZ8xIUNZHeaQfuS71B2"  # I can't create my model in app i buy credits but i can't use it also
        #     "iI0Fh6sIVVZjRWJiecLipOuAYIi2"
        #     # when I push the swap button the App goes back to the extract page. I have tried to re instal the App but still the same, I have deleted my model and created a new, but still the same problem.
        # ]

        read_cost = 0
        user_issues = FirestoreFunctions.db.collection(u'userIssues').stream()

        user_issues_dict = {}
        for doc in user_issues:
            user_id = doc.get("userId")

            # if user_id not in promo_dict.keys() and user_id not in user_issue_ids_do_not_delete:
            #     # user should be < version 58 and no promo
            #     user_doc = FirestoreFunctions.db.collection(u'users').document(user_id).get()
            #     try:
            #         version = int(user_doc.get("currentVersionCode"))
            #     except KeyError:
            #         version = 0
            #     has_purchase = user_doc.get("hasPurchase")
            #     if version < 58 and not has_purchase:
            #         # give 200 credits, and send notification, and delete the issue
            #         FirestoreFunctions.send_notification(user_id,
            #                                              "Create model bug fix! Update version to 2.1.10")
            #         FirestoreFunctions.db.collection(u'userIssues').document(doc.id).delete()
            #
            #         FirestoreFunctions.db.collection(u'users').document(user_id).set({
            #             'credits': google.cloud.firestore_v1.Increment(200)
            #             # Replace 'field_name_to_increment' and 5 with your field and increment value
            #         }, merge=True)
            #         print(f"Issued 200 credits to {user_id}")

            message = doc.get("content")
            submitted_time = doc.get("published")
            solved = doc.get("solved")

            read_cost += 1
            user_issues_dict[submitted_time] = {
                "user_id": user_id,
                'issue_id': doc.id,
                "message": message,
                "solved": solved
            }

            print(f"\rissue:{doc.id}, user_id:{user_id}, submitted_time:{submitted_time}", end="")


            user_messages = FirestoreFunctions.db.collection(u'userIssues').document(doc.id).collection('messages').stream()
            for user_message in user_messages:
                message = user_message.get("content")
                submitted_time = user_message.get("published")
                message_user_id = user_message.get("userId")

                read_cost += 1
                user_issues_dict[submitted_time] = {
                    "user_id": message_user_id,
                    'issue_id': user_message.id,
                    "message": message,
                    "solved": solved
                }

                print(f"\rissue:{user_message.id}, user_id:{message_user_id}, submitted_time:{submitted_time}", end="")


        print("\rsorted-------------")

        user_issues_dict = FirestoreFunctions.sort_dict(user_issues_dict)
        for key,value in user_issues_dict.items():
            print(f"[solved={value['solved']}]", "issue:", value['issue_id'], "user:", value['user_id'], FirestoreFunctions.human_time(key)[1])
            print(value['message'])
            print("-------------")

        print("read_cost/issues", read_cost)

    @staticmethod
    def sort_dict(the_dict):
        tmp_list = []

        items_to_add = len(the_dict)

        for key,value in the_dict.items():
            tmp_list.append(key)

        tmp_list.sort(reverse=True)

        dict_return = {}
        for value in tmp_list:
            dict_return[value] = the_dict[value]
        return dict_return

    @staticmethod
    def delete_user_issues(user_issue_ids):
        for issue_id in user_issue_ids:
            FirestoreFunctions.db.collection(u'userIssues').document(issue_id).delete()

    @staticmethod
    def get_old_jobs():
        read_cost = 0

        docs = FirestoreFunctions.db.collection_group(u'jobs').where('jobStatus', 'in', ['started', 'queued']).order_by('queuedTime', direction=firestore.firestore.Query.ASCENDING).stream()

        for doc in docs:
            read_cost += 1

            try:
                userIp = doc.get("userIp")
                userId = doc.get("userId")
            except KeyError:
                continue

            jobStatus = doc.get("jobStatus")
            jobType = doc.get("jobType")
            uploadType = doc.get("uploadType")
            credits_purchased = 0


            # get user_purchases and current credits

            user_purchases = FirestoreFunctions.db.collection(u'users').document(userId).collection(
                u'purchases').stream()

            for user_purchase in user_purchases:
                read_cost += 1

                if user_purchase.get('processed') is True:
                    sku = user_purchase.get("sku")
                    if sku == 'dfstudio_credits_100':
                        credits_purchased += 100
                    elif sku == 'dfstudio_credits_300':
                        credits_purchased += 300
                    elif sku == 'dfstudio_credits_1000':
                        credits_purchased += 1000

            user_doc = FirestoreFunctions.db.collection(u'users').document(userId).get()
            read_cost += 1
            current_credits = user_doc.get("credits")
            current_version = user_doc.get("currentVersionCode")

            print(f"{doc.id}, userId:{userId}, [{current_version}]credits_purchased:{credits_purchased}, current_credits:{current_credits} [{jobStatus}], jobType:{jobType}, uploadType:{uploadType}")

            # if int(current_version) == 43 and credits_purchased == 300:
            #     FirestoreFunctions.db.collection(u'users').document(userId).collection("jobs").document(doc.id).delete()
            #     FirestoreFunctions.db.collection("jobs").document(doc.id).delete()
            #     FirestoreFunctions.db.collection(u'users').document(userId).set(
            #         {"credits": 0},
            #         merge=True
            #     )
            #     print(f"fixed user:{userId} and deleted job:{doc.id}")

        print("read_cost", read_cost)

    # Function to delete all files in a folder
    @staticmethod
    def delete_folder(folder_path):
        blobs = FirestoreFunctions.bucket.list_blobs(prefix=folder_path)

        for blob in blobs:
            blob.delete()
            print(f'Deleted {blob.name}')

    @staticmethod
    def delete_item(folder_path, item_name):
        blobs = FirestoreFunctions.bucket.list_blobs(prefix=folder_path)

        for blob in blobs:
            if blob.name == item_name:
                blob.delete()
                print(f'Deleted {blob.name}')
                return True

        print(f'Could not find {item_name} in {folder_path}')
        return False

    @staticmethod
    def remove_expired_models(users_root_dir, deleted_models_only, expire_days):
        models_to_remove = 0
        for user_id in [x for x in os.listdir(users_root_dir) if os.path.isdir(os.path.join(users_root_dir, x))]:
            model_root_dir = os.path.join(users_root_dir, user_id, "faceModels")
            for model_id in [x for x in os.listdir(model_root_dir) if os.path.isdir(os.path.join(model_root_dir, x))]:
                model_dir = os.path.join(model_root_dir, model_id)
                model_files = os.listdir(model_dir)

                if deleted_models_only:
                    # check firebase for model existence then delete from local and firebase storage
                    model_doc = FirestoreFunctions.usersRef.document(user_id).collection('faceModels').document(model_id).get()
                    if model_doc.exists:
                        continue
                    else:
                        file_modified_time = os.path.getmtime(model_dir)
                        print(
                            f"{user_id}, {model_id}: file_modified_time={file_modified_time}, {((time.time() - file_modified_time) / 60 / 60 / 24):.02f} days")
                        models_to_remove += 1
                        FirestoreFunctions.delete_folder(f'users/{user_id}/faceModels/{model_id}')
                        shutil.rmtree(model_dir)

                        continue

                # ignore if trained 1000 iters or less; they will have at least 24(4 + 20 every 50 iters) images and 1 checkpoint
                if len(model_files) >= 20:
                    continue
                swapped_dir = os.path.join(model_dir, "swapped")
                # check if more than the expired time

                if os.path.exists(swapped_dir):
                    file_modified_time = os.path.getmtime(swapped_dir)
                    if os.path.getmtime(model_dir) > file_modified_time:
                        file_modified_time = os.path.getmtime(model_dir)
                else:
                    file_modified_time = os.path.getmtime(model_dir)

                days = ((time.time() - file_modified_time) / 60 / 60 / 24)
                if days >= expire_days:
                    print(f"{user_id}, {model_id}: file_modified_time={file_modified_time}, {((time.time() - file_modified_time) / 60 / 60 / 24):.02f} days")
                    modelref = FirestoreFunctions.usersRef.document(user_id).collection('faceModels').document(
                        model_id)
                    model_doc = modelref.get()

                    if model_doc.exists:
                        # reset the model to zero iters if under 1000
                        model_iterations = model_doc.get("iterations")
                        if model_iterations < 1000:
                            modelref.set({
                                'iterations': 0,
                            }, merge=True)
                            print(f"{model_id}: iterations={model_iterations}, expired, setting iterations to zero")
                            # delete the checkpoint to save space
                            FirestoreFunctions.delete_item(f'users/{user_id}/faceModels/{model_id}', 'dfsModel.ckpt')
                    else:
                        # model deleted by user, delete everything from storage
                        FirestoreFunctions.delete_folder(f'users/{user_id}/faceModels/{model_id}')

                    # delete local files
                    shutil.rmtree(model_dir)
                    models_to_remove += 1
                # checkpoint_file = os.path.join(model_dir, "dfsModel.ckpt")
                # if os.path.exists(checkpoint_file):
                #     file_modified_time = os.path.getmtime(checkpoint_file)
                #     print(f"{user_id}, {model_id}: file_modified_time={file_modified_time}, {((time.time() - file_modified_time)/60/60/24):.02f} days")
        print(f"models_to_remove={models_to_remove}")

    @staticmethod
    def remove_swapped_files(users_root_dir):
        files_removed = 0
        for user_id in [x for x in os.listdir(users_root_dir) if os.path.isdir(os.path.join(users_root_dir, x))]:
            model_root_dir = os.path.join(users_root_dir, user_id, "faceModels")
            for model_id in [x for x in os.listdir(model_root_dir) if os.path.isdir(os.path.join(model_root_dir, x))]:
                model_dir = os.path.join(model_root_dir, model_id)

                swapped_dir = os.path.join(model_dir, "swapped")
                # check if more than the expired time
                if os.path.exists(swapped_dir):
                    shutil.rmtree(swapped_dir)
                    files_removed += 1

        print(f"files removed: {files_removed}")
