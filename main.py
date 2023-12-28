import functions_framework
import os
import uuid
from datetime import datetime
import json
from google.cloud import aiplatform


PROJECT_ID = "ameai-causal"
REGION = "asia-east1"
BUCKET_URI = "gs://causal_data"

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
    
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    JOB_DISPLAY_NAME = "pytorch-custom-job"
    TRAIN_IMAGE_URI = "asia.gcr.io/ameai-causal/causal-causica@sha256:46f7cb587d8cb5ebf43578437724e3d967defdbcec69b89eea8407d479921d4c"

    job = aiplatform.CustomContainerTrainingJob(
        display_name=JOB_DISPLAY_NAME,
        container_uri=TRAIN_IMAGE_URI
    )
    
    model_id = uuid.uuid4()
    args = ["--data_csv",
            user_input["data"],
            "--model-id",
            model_id,
            ]
    
    if "dtype" in user_input:
        args.append("--dtype")
        args.appned(user_input["dtype"])
    if "constraint" in user_input:
        args.append("--constraint")
        args.appned(user_input["constraint"])
        
    job.run(
        replica_count=1,
        machine_type="n1-standard-4",
        args=args
    )

    # Note: the total function invocation count across
    # all instances may not be equal to this value!
    return {"model_id": model_id}
    