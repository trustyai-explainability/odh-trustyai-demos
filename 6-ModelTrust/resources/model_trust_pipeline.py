import os
import kfp
import kfp_tekton
from typing import Literal
from dotenv import load_dotenv
from kubernetes.client.models import V1EnvVar

"""
Model Trust Functions for Data Science Pipeline
"""


def load_calibration_data(
    x_cal_file: kfp.components.OutputPath(),
    y_cal_file: kfp.components.OutputPath(),
):
    print("Initializing task to download calibration data...")
    # imports
    import os
    import boto3
    import pickle
    import warnings
    import pandas as pd

    warnings.filterwarnings("ignore")

    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_endpoint_url = os.getenv("AWS_ENDPOINT_URL")
    aws_region_name = os.getenv("AWS_REGION")

    data_bucket = os.getenv("DATA_BUCKET")  # s3 bucket with calibration data
    calibration_data_x_path = os.getenv("CALIBRATION_DATA_X_PATH")
    calibration_data_y_path = os.getenv("CALIBRATION_DATA_Y_PATH")

    # s3 connection
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,  # os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=aws_secret_access_key,  # os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    s3_client = session.client(
        "s3",
        endpoint_url=aws_endpoint_url,  # os.getenv("AWS_ENDPOINT_URL"),
        region_name=aws_region_name,  # os.getenv("AWS_REGION"),
    )

    x_cal = pd.read_csv(
        s3_client.get_object(Bucket=data_bucket, Key=calibration_data_x_path)["Body"]
    )
    print("completed downloading x_cal.")

    y_cal = pd.read_csv(
        s3_client.get_object(Bucket=data_bucket, Key=calibration_data_y_path)["Body"]
    )
    print("completed downloading y_cal.")

    x_cal = x_cal.values
    y_cal = y_cal.values.flatten()

    def save_pickle(object_file, target_object):
        with open(object_file, "wb") as f:
            pickle.dump(target_object, f)

    save_pickle(x_cal_file, x_cal)
    save_pickle(y_cal_file, y_cal)

    print("Completed persisting calibration data.")


def load_base_model(
    base_model_file: kfp.components.OutputPath(),
):
    print("Initializing task to download base model...")
    # imports
    import os
    import boto3
    import pickle
    import warnings

    warnings.filterwarnings("ignore")

    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_endpoint_url = os.getenv("AWS_ENDPOINT_URL")
    aws_region_name = os.getenv("AWS_REGION")

    model_bucket = os.getenv("MODEL_BUCKET")  # s3 bucket with calibration data
    base_model_path = os.getenv("BASE_MODEL_PATH")

    # s3 connection
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,  # os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=aws_secret_access_key,  # os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    s3_client = session.client(
        "s3",
        endpoint_url=aws_endpoint_url,  # os.getenv("AWS_ENDPOINT_URL"),
        region_name=aws_region_name,  # os.getenv("AWS_REGION"),
    )

    base_onnx_model_str = s3_client.get_object(
        Bucket=model_bucket, Key=base_model_path
    )["Body"].read()

    print("completed downloading base model.")

    def save_pickle(object_file, target_object):
        with open(object_file, "wb") as f:
            pickle.dump(target_object, f)

    save_pickle(base_model_file, base_onnx_model_str)
    print("Completed persisting base model.")


def train_model_trust_model(
    x_cal_file: kfp.components.InputPath(),
    y_cal_file: kfp.components.InputPath(),
    base_model_file: kfp.components.InputPath(),
    model_trust_wrapped_model_path_file: kfp.components.OutputTextFile(),
):
    print("Initializing task to train model trust model...")

    # imports
    import os
    import random
    import string
    import boto3
    import pickle
    import warnings
    from model_trust.regression.region_uncertainty_estimation import (
        RegionUncertaintyEstimator,
    )

    warnings.filterwarnings("ignore")

    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_endpoint_url = os.getenv("AWS_ENDPOINT_URL")
    aws_region_name = os.getenv("AWS_REGION")

    model_bucket = os.getenv("MODEL_BUCKET")  # s3 bucket with calibration data

    # model trust parameters
    model_trust_params = {}
    model_trust_params["regions_model"] = os.getenv("REGION_TYPE")
    model_trust_params["confidence"] = int(os.getenv("CONFIDENCE"))

    if model_trust_params["regions_model"] not in ["single_region", "multi_region"]:
        raise Exception(
            "Model Trust region model type {} is not supported. Use 'single_region' or 'multi_region'.".format(
                model_trust_params["regions_model"]
            )
        )

    if model_trust_params["regions_model"] == "multi_region":
        model_trust_params["multi_region_model_selection_metric"] = os.getenv(
            "MULTI_REGION_MODEL_SELECTION_METRIC"
        )
        model_trust_params["multi_region_model_selection_stat"] = os.getenv(
            "MULTI_REGION_MODEL_SELECTION_STAT"
        )
        model_trust_params["multi_region_min_group_size"] = int(
            os.getenv("MULTI_REGION_MIN_GROUP_SIZE")
        )

    def load_pickle(object_file):
        with open(object_file, "rb") as f:
            target_object = pickle.load(f)
        return target_object

    x_cal = load_pickle(x_cal_file)
    y_cal = load_pickle(y_cal_file)
    print("loaded calibration data.")

    base_onnx_model_str = load_pickle(base_model_file)
    print("loaded base model.")

    model_trust_params["base_model"] = base_onnx_model_str

    multi_region_cp_model = RegionUncertaintyEstimator(**model_trust_params)
    print("initialized Region Uncertainty Estimator.")

    multi_region_cp_model.fit(x_cal, y_cal)
    print("trained Region Uncertainty Estimator.")

    model_trust_wrapped_model = multi_region_cp_model.export_learned_config()[
        "combined_model"
    ]
    print("retrieved wrapped model trust model.")

    # s3 connection
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,  # os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=aws_secret_access_key,  # os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    s3_client = session.client(
        "s3",
        endpoint_url=aws_endpoint_url,  # os.getenv("AWS_ENDPOINT_URL"),
        region_name=aws_region_name,  # os.getenv("AWS_REGION"),
    )

    # s3 path for model trust model
    model_trust_wrapped_model_path = (
        "onnx_models/{}_model_trust_ds_model_{}.onnx".format(
            model_trust_params["regions_model"],
            "".join(random.choices(string.ascii_lowercase + string.digits, k=5)),
        )
    )
    s3_client.put_object(
        Body=model_trust_wrapped_model,
        Bucket=model_bucket,
        Key=model_trust_wrapped_model_path,
    )
    model_trust_wrapped_model_path_file.write(model_trust_wrapped_model_path)
    print("Completed uploading model trust model to s3.")


def deploy_model_trust_model(
    model_trust_wrapped_model_path_file: kfp.components.InputTextFile(),
    model_trust_service_name_file: kfp.components.OutputTextFile(),
):
    print("Initializing task to deploy model trust model...")

    # imports
    import os
    import warnings
    from kubernetes import client
    from kserve import KServeClient
    from kserve import constants
    from kserve import V1beta1PredictorSpec, V1beta1Batcher
    from kserve import V1beta1ModelSpec, V1beta1ModelFormat
    from kserve import V1beta1StorageSpec
    from kserve import V1beta1InferenceServiceSpec
    from kserve import V1beta1InferenceService

    warnings.filterwarnings("ignore")

    model_trust_wrapped_model_path = model_trust_wrapped_model_path_file.readline()
    service_name = model_trust_wrapped_model_path.split("/")[-1].split(".onnx")[0]
    service_name = service_name.replace("_", "-")

    oc_project_name = os.getenv("ODH_DATA_SCIENCE_PROJECT")
    s3_resource_name = os.getenv("ODH_DATA_CONNECTION_NAME")
    oc_host = os.getenv("OC_HOST")
    oc_user = os.getenv("OC_USER")
    oc_token = os.getenv("OC_TOKEN")

    def _prepare_kube_config(oc_host, oc_user, oc_token, oc_project):
        kube_config_dict = {
            "kind": "Config",
            "apiVersion": "v1",
            "preferences": {},
            "clusters": [
                {
                    "cluster": {"insecure-skip-tls-verify": True, "server": ""},
                    "name": "",
                }
            ],
            "users": [{"name": "", "user": {"token": ""}}],
            "contexts": [
                {"name": "", "context": {"cluster": "", "namespace": "", "user": ""}}
            ],
            "current-context": "",
        }

        kube_config_dict["clusters"][0]["cluster"]["server"] = oc_host
        kube_config_dict["clusters"][0]["name"] = oc_host.split("https://")[-1]

        kube_config_dict["users"][0]["name"] = (
            oc_user + "/" + kube_config_dict["clusters"][0]["name"]
        )
        kube_config_dict["users"][0]["user"]["token"] = oc_token

        kube_config_dict["contexts"][0]["name"] = (
            oc_project + "/" + kube_config_dict["clusters"][0]["name"] + "/" + oc_user
        )
        kube_config_dict["contexts"][0]["context"]["cluster"] = kube_config_dict[
            "clusters"
        ][0]["name"]
        kube_config_dict["contexts"][0]["context"]["namespace"] = oc_project
        kube_config_dict["contexts"][0]["context"]["user"] = kube_config_dict["users"][
            0
        ]["name"]

        kube_config_dict["current-context"] = kube_config_dict["contexts"][0]["name"]
        return kube_config_dict

    kube_config_dict = _prepare_kube_config(
        oc_host, oc_user=oc_user, oc_token=oc_token, oc_project=oc_project_name
    )

    default_model_spec = V1beta1InferenceServiceSpec(
        predictor=V1beta1PredictorSpec(
            model=V1beta1ModelSpec(
                model_format=V1beta1ModelFormat(name="onnx"),
                runtime="triton-2.x",
                storage=V1beta1StorageSpec(
                    key=s3_resource_name, path=model_trust_wrapped_model_path
                ),
            ),
            batcher=V1beta1Batcher(max_batch_size=100),
        )
    )

    isvc = V1beta1InferenceService(
        api_version=constants.KSERVE_V1BETA1,
        kind=constants.KSERVE_KIND,
        metadata=client.V1ObjectMeta(
            name=service_name,
            namespace=oc_project_name,
            annotations={"serving.kserve.io/deploymentMode": "ModelMesh"},
        ),
        spec=default_model_spec,
    )

    kserve_client = KServeClient(config_dict=kube_config_dict)

    # create inference service
    kserve_client.create(isvc)
    model_trust_service_name_file.write(service_name)
    print(
        "Completed deploying model trust model {} on Model Mesh...".format(service_name)
    )


"""
Model Trust ContainerOps/Steps in Data Science Pipeline
"""

load_calibration_data_op = kfp.components.create_component_from_func(
    load_calibration_data,
    base_image="registry.access.redhat.com/ubi8/python-39",
    packages_to_install=["boto3", "pandas"],
)

load_base_model_op = kfp.components.create_component_from_func(
    load_base_model,
    base_image="registry.access.redhat.com/ubi8/python-39",
    packages_to_install=["boto3", "pandas"],
)

# model_trust[deploy] @ git+https://github.com/trustyai-explainability/trustyai-model-trust.git
train_model_trust_model_op = kfp.components.create_component_from_func(
    train_model_trust_model,
    base_image="registry.access.redhat.com/ubi8/python-39",
    packages_to_install=[
        "boto3",
        "pandas",
        "model_trust @ git+https://github.com/trustyai-explainability/trustyai-model-trust.git",
    ],
)
deploy_model_trust_model_op = kfp.components.create_component_from_func(
    deploy_model_trust_model,
    base_image="registry.access.redhat.com/ubi8/python-39",
    packages_to_install=[
        "boto3",
        "kserve",
    ],
)


"""
Construct Data Science Pipeline.
"""


@kfp.dsl.pipeline(
    name="Model Trust Pipeline",
)
def model_trust_pipeline(
    aws_access_key: str,
    aws_secret: str,
    oc_host: str,
    oc_token: str,
    s3_endpoint: str = "https://s3.us.cloud-object-storage.appdomain.cloud",
    s3_region: str = "us-geo",
    data_bucket: str = "model-trust",
    calibration_data_x_path: str = "simulated_data/x_cal.csv",
    calibration_data_y_path: str = "simulated_data/y_cal.csv",
    model_bucket: str = "model-trust",
    base_model_path: str = "onnx_models/base_onnx_model.onnx",
    oc_user: str = "kube:admin",
    odh_ds_project_name: str = "model-trust-ds-project",
    odh_data_connection_name: str = "aws-connection-samples3",
    region_type: Literal["single_region", "multi_region"] = "multi_region",
    confidence: int = 95,
    multi_region_model_selection_metric: str = "coverage_ratio",
    multi_region_model_selection_stat: str = "min",
    multi_region_min_group_size: int = 20,
):
    # prepare env variables for model trust ds pipeline
    # AWS S3 credentials
    aws_access_key_var = V1EnvVar(name="AWS_ACCESS_KEY_ID", value=aws_access_key)
    aws_secret_var = V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value=aws_secret)
    aws_endpoint_var = V1EnvVar(name="AWS_ENDPOINT_URL", value=s3_endpoint)
    aws_region_var = V1EnvVar(name="AWS_REGION", value=s3_region)
    # aws_access_key = V1EnvVar(
    #     name="AWS_ACCESS_KEY_ID", value=os.getenv("AWS_ACCESS_KEY_ID")
    # )
    # aws_secret = V1EnvVar(
    #     name="AWS_SECRET_ACCESS_KEY", value=os.getenv("AWS_SECRET_ACCESS_KEY")
    # )
    # aws_endpoint = V1EnvVar(
    #     name="AWS_ENDPOINT_URL", value=os.getenv("AWS_ENDPOINT_URL")
    # )
    # aws_region = V1EnvVar(name="AWS_REGION", value=os.getenv("AWS_REGION"))

    # OpenShift Credentials
    oc_host_var = V1EnvVar(name="OC_HOST", value=oc_host)
    oc_user_var = V1EnvVar(name="OC_USER", value=oc_user)
    oc_token_var = V1EnvVar(name="OC_TOKEN", value=oc_token)

    # oc_host = V1EnvVar(name="OC_HOST", value=os.getenv("OC_HOST"))
    # oc_user = V1EnvVar(name="OC_USER", value=os.getenv("OC_USER"))
    # oc_token = V1EnvVar(name="OC_TOKEN", value=os.getenv("OC_TOKEN"))

    # Open Data Hub Data Science Project / OpenShift Project to deploy the Model Trust model
    odh_ds_project_name_var = V1EnvVar(
        name="ODH_DATA_SCIENCE_PROJECT", value=odh_ds_project_name
    )

    # odh_ds_project_name = V1EnvVar(
    #     name="ODH_DATA_SCIENCE_PROJECT",
    #     value=os.getenv("ODH_DATA_SCIENCE_PROJECT"),
    # )

    # Open Data Hub Data Connection to retrieve the Model Trust model for deployment
    odh_data_connection_name_var = V1EnvVar(
        name="ODH_DATA_CONNECTION_NAME", value=odh_data_connection_name
    )
    # odh_data_connection_name = V1EnvVar(
    #     name="ODH_DATA_CONNECTION_NAME", value=os.getenv("ODH_DATA_CONNECTION_NAME")
    # )

    # AWS S3 Bucket and File Path for Calibration Data
    data_bucket_var = V1EnvVar(name="DATA_BUCKET", value=data_bucket)
    calibration_data_x_path_var = V1EnvVar(
        name="CALIBRATION_DATA_X_PATH", value=calibration_data_x_path
    )
    calibration_data_y_path_var = V1EnvVar(
        name="CALIBRATION_DATA_Y_PATH", value=calibration_data_y_path
    )

    # data_bucket = V1EnvVar(name="DATA_BUCKET", value=os.getenv("DATA_BUCKET"))
    # calibration_data_x_path = V1EnvVar(
    #     name="CALIBRATION_DATA_X_PATH", value=os.getenv("CALIBRATION_DATA_X_PATH")
    # )
    # calibration_data_y_path = V1EnvVar(
    #     name="CALIBRATION_DATA_Y_PATH", value=os.getenv("CALIBRATION_DATA_Y_PATH")
    # )

    # AWS S3 Bucket and File Path for Base Regression Model
    model_bucket_var = V1EnvVar(name="MODEL_BUCKET", value=model_bucket)
    base_model_path_var = V1EnvVar(name="BASE_MODEL_PATH", value=base_model_path)

    # model_bucket = V1EnvVar(name="MODEL_BUCKET", value=os.getenv("MODEL_BUCKET"))
    # base_model_path = V1EnvVar(
    #     name="BASE_MODEL_PATH", value=os.getenv("BASE_MODEL_PATH")
    # )

    # Pipeline input parameters for every run/experiment.
    region_type_var = V1EnvVar(name="REGION_TYPE", value=region_type)
    confidence_var = V1EnvVar(name="CONFIDENCE", value=confidence)
    multi_region_model_selection_metric_var = V1EnvVar(
        name="MULTI_REGION_MODEL_SELECTION_METRIC",
        value=multi_region_model_selection_metric,
    )
    multi_region_model_selection_stat_var = V1EnvVar(
        name="MULTI_REGION_MODEL_SELECTION_STAT",
        value=multi_region_model_selection_stat,
    )
    multi_region_min_group_size_var = V1EnvVar(
        name="MULTI_REGION_MIN_GROUP_SIZE", value=multi_region_min_group_size
    )

    # task to load data
    load_calibration_data_task = load_calibration_data_op()

    ## set env variables
    load_calibration_data_task.add_env_variable(aws_access_key_var)
    load_calibration_data_task.add_env_variable(aws_secret_var)
    load_calibration_data_task.add_env_variable(aws_endpoint_var)
    load_calibration_data_task.add_env_variable(aws_region_var)
    load_calibration_data_task.add_env_variable(data_bucket_var)
    load_calibration_data_task.add_env_variable(calibration_data_x_path_var)
    load_calibration_data_task.add_env_variable(calibration_data_y_path_var)
    ## set resources
    load_calibration_data_task.set_memory_request("1G")
    load_calibration_data_task.set_memory_limit("2G")
    load_calibration_data_task.set_cpu_request("1")
    load_calibration_data_task.set_cpu_limit("2")

    # task to load model
    load_base_model_task = load_base_model_op()
    ## set env variables
    load_base_model_task.add_env_variable(aws_access_key_var)
    load_base_model_task.add_env_variable(aws_secret_var)
    load_base_model_task.add_env_variable(aws_endpoint_var)
    load_base_model_task.add_env_variable(aws_region_var)
    load_base_model_task.add_env_variable(model_bucket_var)
    load_base_model_task.add_env_variable(base_model_path_var)
    ## set resources
    load_base_model_task.set_memory_request("1G")
    load_base_model_task.set_memory_limit("2G")
    load_base_model_task.set_cpu_request("1")
    load_base_model_task.set_cpu_limit("2")

    # task to train model trust model
    train_model_trust_model_task = train_model_trust_model_op(
        load_calibration_data_task.outputs["x_cal"],
        load_calibration_data_task.outputs["y_cal"],
        load_base_model_task.outputs["base_model"],
    )
    ## set env variables
    train_model_trust_model_task.add_env_variable(aws_access_key_var)
    train_model_trust_model_task.add_env_variable(aws_secret_var)
    train_model_trust_model_task.add_env_variable(aws_endpoint_var)
    train_model_trust_model_task.add_env_variable(aws_region_var)

    train_model_trust_model_task.add_env_variable(model_bucket_var)

    train_model_trust_model_task.add_env_variable(region_type_var)
    train_model_trust_model_task.add_env_variable(confidence_var)
    train_model_trust_model_task.add_env_variable(
        multi_region_model_selection_metric_var
    )
    train_model_trust_model_task.add_env_variable(multi_region_model_selection_stat_var)
    train_model_trust_model_task.add_env_variable(multi_region_min_group_size_var)
    ## set resources
    training_memory_request = os.getenv("TRAINING_MEMORY_REQUEST", "2G")
    training_memory_limit = os.getenv("TRAINING_MEMORY_LIMIT", "4G")
    training_cpu_request = os.getenv("TRAINING_CPU_REQUEST", "1")
    training_cpu_limit = os.getenv("TRAINING_CPU_LIMIT", "2")

    train_model_trust_model_task.set_memory_request(training_memory_request)
    train_model_trust_model_task.set_memory_limit(training_memory_limit)
    train_model_trust_model_task.set_cpu_request(training_cpu_request)
    train_model_trust_model_task.set_cpu_limit(training_cpu_limit)

    # task to deploy model trust model
    deploy_model_trust_model_task = deploy_model_trust_model_op(
        train_model_trust_model_task.outputs["model_trust_wrapped_model_path"],
    )
    ## set env variables
    deploy_model_trust_model_task.add_env_variable(odh_ds_project_name_var)
    deploy_model_trust_model_task.add_env_variable(odh_data_connection_name_var)
    deploy_model_trust_model_task.add_env_variable(oc_host_var)
    deploy_model_trust_model_task.add_env_variable(oc_user_var)
    deploy_model_trust_model_task.add_env_variable(oc_token_var)


"""
Additional steps for Fyre clusters.
"""


def route_image_request_to_remote_artifactory(filename):
    import yaml

    with open(filename) as f:
        list_doc = yaml.safe_load(f)

    list_doc["spec"]["serviceAccountName"] = "model-trust-workbench"
    for task in list_doc["spec"]["pipelineSpec"]["tasks"]:
        for step in task["taskSpec"]["steps"]:
            if step["image"] == "busybox":
                step[
                    "image"
                ] = "docker-na.artifactory.swg-devops.com/res-srom-docker-remote/busybox"

    with open(filename, "w") as f:
        yaml.dump(list_doc, f)


if __name__ == "__main__":
    # load environment variables from .env file.
    dirpath = os.path.dirname(os.path.realpath(__file__))
    env_filepath = os.path.realpath(os.path.join(dirpath, "./.env"))
    if os.path.exists(env_filepath):
        load_dotenv(env_filepath, override=True)

    os.environ["DEFAULT_STORAGE_CLASS"] = os.getenv(
        "DEFAULT_STORAGE_CLASS", "rook-cephfs"
    )
    os.environ["DEFAULT_ACCESSMODES"] = os.getenv(
        "DEFAULT_ACCESSMODES", "ReadWriteOnce"
    )

    yaml_file_name = __file__.replace(".py", ".yaml")

    kfp_tekton.compiler.TektonCompiler().compile(model_trust_pipeline, yaml_file_name)

    # route the image requests to remote artifactory repository to avoid docker rate limit issues.
    route_image_request_to_remote_artifactory(yaml_file_name)
