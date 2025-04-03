import datetime
import time

from firebase_admin import firestore
from google.cloud.firestore_v1 import FieldFilter

from fire_functions import FirestoreFunctions
firebase = FirestoreFunctions()


def get_active_servers(active_time):
    active_servers = []

    # Get current timestamp
    current_time = int(datetime.datetime.utcnow().timestamp())

    docs = firebase.db.collection("repainterServers").get()
    for doc in docs:
        timestamp = doc.get("lastActiveTime")  # Firestore stores timestamps as datetime
        unix_timestamp = int(timestamp.timestamp())  # Convert to Unix timestamp (seconds)
        if current_time - unix_timestamp <= active_time:
            active_servers.append(doc)

    return active_servers


def run(active_time, estimated_time_per_job):
    # get all server docs less than active_time
    active_servers = get_active_servers(active_time=active_time)

    # get all active and queued jobs
    active_jobs = FirestoreFunctions.repaintImageJobsRef.where(
        filter=FieldFilter("jobStatus", "==", "active")) \
        .order_by(u'queuedTime', direction=firestore.firestore.Query.ASCENDING).get()

    queued_jobs = FirestoreFunctions.repaintImageJobsRef.where(
        filter=FieldFilter("jobStatus", "==", "queued")) \
        .order_by(u'queuedTime', direction=firestore.firestore.Query.ASCENDING).get()

    queue_times = []

    free_servers = len(active_servers) - len(active_jobs)

    # account estimated time split between servers
    estimated_time_per_job = estimated_time_per_job / max(len(active_servers), 1)

    for index, queued_job in enumerate(queued_jobs):
        # this ensures we minus time according to active servers
        estimated_time = max(
            0,
            estimated_time_per_job +
            (index*estimated_time_per_job) -
            (free_servers*estimated_time_per_job)
        )
        data = {
            "jobId": queued_job.id,
            "estimatedSeconds": estimated_time
        }
        queue_times.append(data)
        print(data)

    # add active servers
    firebase.db.collection("repainterQueues").document("liveQueue").set(
        {"queueTimes": queue_times, "activeServers": len(active_servers)}
    )


if __name__ == "__main__":
    while True:
        run(active_time=100, estimated_time_per_job=40)
        print("sleeping 5...")
        time.sleep(5)
