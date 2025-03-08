import os

api_key = os.environ.get("DFS_CRED_type")
if api_key is None:
    print("API_KEY environment variable not set")
else:
    print(f"API_KEY: {api_key}")