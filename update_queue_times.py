import datetime
import time

from firebase_admin import firestore
from google.cloud.firestore_v1 import FieldFilter

from fire_functions import FirestoreFunctions
firebase = FirestoreFunctions()


def get_active_servers(active_time):
    active_servers = []
    free_servers = []
    # Get current timestamp
    current_time = datetime.datetime.now(datetime.timezone.utc).timestamp()

    docs = firebase.db.collection("repainterServers").get()
    for doc in docs:
        timestamp = doc.get("lastActiveTime")  # Firestore stores timestamps as datetime
        is_busy = bool(doc.get("isBusy"))  # Firestore stores timestamps as datetime
        unix_timestamp = int(timestamp.astimezone(datetime.timezone.utc).timestamp())  # Convert to Unix timestamp (seconds)

        if current_time - unix_timestamp <= active_time:
            active_servers.append(doc)
            if not is_busy:
                free_servers.append(doc)

    return active_servers, free_servers


def run(active_time, estimated_time_per_job):
    # get all server docs less than active_time
    active_servers, free_servers = get_active_servers(active_time=active_time)

    # get all active and queued jobs
    active_jobs = FirestoreFunctions.repaintImageJobsRef.where(
        filter=FieldFilter("jobStatus", "==", "active")) \
        .order_by(u'queuedTime', direction=firestore.firestore.Query.ASCENDING).get()

    queued_jobs = FirestoreFunctions.repaintImageJobsRef.where(
        filter=FieldFilter("jobStatus", "==", "queued")) \
        .order_by(u'queuedTime', direction=firestore.firestore.Query.ASCENDING).get()

    queue_times = []

    # account estimated time split between servers
    estimated_time_per_job_per_server = estimated_time_per_job / max(len(active_servers), 1)

    for index, queued_job in enumerate(queued_jobs):
        # this ensures we minus time according to active servers
        if index == 0:
            estimated_time = estimated_time_per_job
        else:
            estimated_time = max(
                0,
                estimated_time_per_job_per_server +
                (index*estimated_time_per_job_per_server) -
                (len(free_servers)*estimated_time_per_job_per_server)
            )
        data = {
            "jobId": queued_job.id,
            "estimatedSeconds": estimated_time
        }
        queue_times.append(data)
        print(data)
    print({"activeServers": len(active_servers)})
    # add active servers
    firebase.db.collection("repainterQueues").document("liveQueue").set(
        {"queueTimes": queue_times, "activeServers": len(active_servers)}
    )


def test_utc_time():
    uct_now_time = datetime.datetime.utcnow()
    utc_time = datetime.datetime.now(datetime.timezone.utc)

    print("utc_time", utc_time.timestamp())
    print("uct_now_time", uct_now_time.timestamp())
    firebase.db.collection("repainterServers").document("test").set(
        {"utc_time": utc_time, "uct_now_time": uct_now_time}
    )

    doc = firebase.db.collection("repainterServers").document("test").get().to_dict()
    doc_utc_time = doc["utc_time"]
    doc_uct_now_time = doc["uct_now_time"]

    print("doc_utc_time", doc_utc_time.timestamp())
    print("doc_uct_now_time", doc_uct_now_time.timestamp())


if __name__ == "__main__":

    while True:
        run(active_time=100, estimated_time_per_job=40)
        print("sleeping 5...")
        time.sleep(5)
