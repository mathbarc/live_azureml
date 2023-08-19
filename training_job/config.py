import os

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv(".env")

# authenticate
credential = DefaultAzureCredential()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=os.environ["SUBSCRIPTION_ID"],
    resource_group_name=os.environ["RESOURCE_GROUP_NAME"],
    workspace_name=os.environ["WORKSPACE_NAME"]
)
