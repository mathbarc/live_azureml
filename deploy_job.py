import os

from config import ml_client

from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command, Input
from azureml.fsspec import AzureMachineLearningFileSystem

# Name assigned to the compute cluster
cpu_compute_target = "cpu-rice-classifier"

try:
    # let's see if the compute target already exists
    cpu_cluster = ml_client.compute.get(cpu_compute_target)
    print(
        f"You already have a cluster named {cpu_compute_target}, we'll reuse it as is."
    )

except Exception:
    print("Creating a new cpu compute target...")

    # Let's create the Azure Machine Learning compute object with the intended parameters
    # if you run into an out of quota error, change the size to a comparable VM that is available.\
    # Learn more on https://azure.microsoft.com/en-us/pricing/details/machine-learning/.
    cpu_cluster = AmlCompute(
        name=cpu_compute_target,
        # Azure Machine Learning Compute is the on-demand VM service
        type="amlcompute",
        # VM Family
        size="Standard_DS11_v2",
        # Minimum running nodes when there is no job running
        min_instances=0,
        # Nodes in cluster
        max_instances=1,
        # How many seconds will the node running after the job termination
        idle_time_before_scale_down=180,
        # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
        tier="Dedicated",
    )
    print(
        f"AMLCompute with name {cpu_cluster.name} will be created, with compute size {cpu_cluster.size}"
    )
    # Now, we pass the object to MLClient's create_or_update method
    cpu_cluster = ml_client.compute.begin_create_or_update(cpu_cluster)
    

filedataset_asset = ml_client.data.get(name="rice_dataset", version="6")

job = command(
    inputs={
        "dataset":Input(type=filedataset_asset.type,path=filedataset_asset.path),
        "lr": 0.0001,
        "momentum": 0.8,
        "batch_size": 12,
        "epochs": 20
    },
    # inputs={
    #     "dataset":Input(type=filedataset_asset.type,path=filedataset_asset.path),
    #     "lr": 0.001,
    #     "momentum": 0.8,
    #     "batch_size": 20,
    #     "epochs": 20
    # },
    code="./training_job/",  # location of source code
    command="python train_rice_classifier.py --dataset ${{inputs.dataset}} --lr ${{inputs.lr}} --momentum ${{inputs.momentum}} --batch_size ${{inputs.batch_size}} --epochs ${{inputs.epochs}}",
    environment="rice-classifier-training-env@latest",
    compute=cpu_compute_target, #delete this line to use serverless compute
    display_name="rice_classifier_training",
    experiment_name="Rice Classifier",
    environment_variables={
                           "SUBSCRIPTION_ID":os.environ["SUBSCRIPTION_ID"], 
                           "RESOURCE_GROUP_NAME":os.environ["RESOURCE_GROUP_NAME"],
                           "WORKSPACE_NAME":os.environ["WORKSPACE_NAME"]
                           }
)

ml_client.create_or_update(job)