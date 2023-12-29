import functions_framework
import os
import uuid
from datetime import datetime
import json
from google.cloud import aiplatform
import torch
from causica.sem.distribution_parameters_sem import DistributionParametersSEM
from tensordict import TensorDict


PROJECT_ID = "ameai-causal"
REGION = "asia-east1"
MODEL_BUCKET_URI = "gs://causal_models"

@functions_framework.http
def inference(request):
    user_input = request.get_json()
    
    model_id = user_input["model_id"]
    
    loaded_model = torch.load(os.path.join(MODEL_BUCKET_URI, model_id) + ".pt")
    
    normalizer = loaded_model["normalizer"]
    graph = loaded_model["new_adg"]
    model = loaded_model["model"]
    
    SEM_MODULE = lightning_module.sem_module()
    sem = DistributionParametersSEM(graph, SEM_MODULE._noise_module, SEM_MODULE._functional_relationships)
    
    before = dict()
    after = dict()
    for key, value in user_input["before"].items():
        before[key] = torch.tensor([value]).float()

    for key, value in user_input["after"].items():
        after[key] = torch.tensor([value]).float()
        
    before = TensorDict(before, batch_size=tuple())
    after = TensorDict(after, batch_size=tuple())
    
    intervention_a = normalizer(before)
    intervention_b = normalizer(after)
    
    torch_shape = torch.Size([20000])
    
    rev_a_samples = normalizer.inv(sem.do(interventions=intervention_a).sample(torch_shape))
    rev_b_samples = normalizer.inv(sem.do(interventions=intervention_b).sample(torch_shape))
    
    output = {}
    for key in rev_a_samples.keys():
        ate_mean = (rev_b_samples[key].mean(0) - rev_a_samples[key].mean(0)).detach().cpu().numpy()[0].item()
        output[key] = ate_mean
    
    return output
    
    
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
    
    aiplatform.init(project=PROJECT_ID, location=REGION)

    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    JOB_DISPLAY_NAME = "pytorch-custom-job"
    TRAIN_IMAGE_URI = os.environ["TRAINER_IMAGE"]

    job = aiplatform.CustomContainerTrainingJob(
        display_name=JOB_DISPLAY_NAME,
        container_uri=TRAIN_IMAGE_URI
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
    