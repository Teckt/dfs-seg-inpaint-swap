#!/usr/bin/env python3
import json
import os
import sys
import boto3
from botocore.exceptions import ClientError


def get_secret():

    secret_name = "FIREBASE_CRED_secret_key"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    secret = get_secret_value_response['SecretString']

    # Your code goes here.


PREFIX = "FIREBASE_CRED_"

def json_to_export_command(json_filepath):
    """
    Read a JSON file and generate a single shell command that exports environment variables.
    Each key from the JSON is prefixed with FIREBASE_CRED_.
    """
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    commands = []
    for key, value in data.items():
        # skip secret as it's too long to be copied
        if key == "private_key":
            continue
        env_key = f"{PREFIX}{key}"
        # Use single quotes around the value to handle spaces/special characters.
        commands.append(f"export {env_key}='{value}'")
    # Combine all export commands into a single command line.
    return "; ".join(commands)

def export_env_to_json(json_filepath):
    """
    Read all environment variables that start with FIREBASE_CRED_
    and save them into a JSON file, removing the prefix.
    """
    data = {}

    secret = get_secret()
    os.environ["FIREBASE_CRED_private_key"] = secret

    for key, value in os.environ.items():
        if key.startswith(PREFIX):
            # Remove the prefix from the key.
            new_key = key[len(PREFIX):]
            data[new_key] = value
    with open(json_filepath, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    # Usage:
    #   python script.py export <input_json>
    #   python script.py save <output_json>
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python script.py export <input_json>")
        print("  python script.py save <output_json>")
        sys.exit(1)

    mode = sys.argv[1]
    filepath = sys.argv[2]

    if mode == "export":
        # Print out a single command to set all variables.
        command_str = json_to_export_command(filepath)
        print(command_str)
    elif mode == "save":
        # Save all FIREBASE_CRED_* env vars into a JSON file (prefix removed).
        export_env_to_json(filepath)
        print(f"Environment variables saved to {filepath}")
    else:
        print("Invalid mode. Use 'export' to generate the export command or 'save' to write to JSON.")
