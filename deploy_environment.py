from config import ml_client
from azure.ai.ml.entities import Environment, BuildContext

model_env = Environment(name="rice-classifier-training-env", 
                        description="Environment to train rice classifier", 
                        build=BuildContext(path="./environment"))

ml_client.environments.create_or_update(model_env)
    
