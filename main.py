import functions_framework
import os
import uuid
from datetime import datetime
import json
from google.cloud import aiplatform
import torch


PROJECT_ID = "ameai-causal"
REGION = "asia-east1"
MODEL_BUCKET_URI = "gs://causal_models" #
STAGING_BUCKET_URI = "gs://causal_models" #causal_data
    
    
@functions_framework.http
def discovery(request):
    """
    HTTP Cloud Function that counts how many times it is executed
    within a specific instance.
    Args:
        request (flask.Request): The request object.
        <http://flask.pocoo.org/docs/1.0/api/#flask.Request>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>.
    """
    
    user_input = request.get_json()
    
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET_URI)

    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    JOB_DISPLAY_NAME = "pytorch-custom-job"
    TRAIN_IMAGE_URI = os.environ["TRAINER_IMAGE"]

    job = aiplatform.CustomContainerTrainingJob(
        display_name=JOB_DISPLAY_NAME,
        container_uri=TRAIN_IMAGE_URI,
    )
    
    model_id = str(uuid.uuid4())
    args = ["--data_csv",
            user_input["data"],
            "--model-id",
            model_id,
            "--model-dir",
            MODEL_BUCKET_URI,
            ]
    
    if "dtype" in user_input:
        args.append("--dtype")
        args.append(user_input["dtype"])
    if "constraint" in user_input:
        args.append("--constraint")
        args.append(user_input["constraint"])
        
    job.run(
        replica_count=1,
        machine_type="n1-standard-4",
        sync=False,
        args=args
    )

    # Note: the total function invocation count across
    # all instances may not be equal to this value!
    return {"model_id": model_id}
    