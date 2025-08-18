"""
This is the script that runs on the "controller" machine: the lightweight 
machine that requests the big boi instance(s) from EC2 and configures the 
training job. In this example, I want the controller machine to be my home
computer. This means that I can run this script from my computer and start a 
training job on the remote big boi instance, without an intermediary machine
from aws.

This script can run in two modes, local and remote, depending on the value of
`run_locally`.

VERY IMPORTANT NOTE: running this script locally is not at all the same as 
running a normal training locally (i.e: running locally the file `train.py`):
Running this script in local mode still creates a sagemaker job and does 
everything in the same exact way as it would to run on a remote instance, 
EXCEPT that the big boi instance in EC2 is replaced by your local machine.

So the script creates a docker container, downloads it, creates the virtual 
environment, and runs the training job in the same exact way as it would on the
remote instance. Running this training job in local mode successfuly should 
GUARANTEE that it will run successfully on a remote instance.
"""

import sagemaker.pytorch
import sagemaker.local
import sagemaker
from datetime import datetime


def launch_aws_job(
    experiment_name: str,
    instance_type: str,
    max_training_hours: int,
    entry_point: str,
    input_mode: str = "File",
    data_dir: str = "",
):
    # True: run training job locally for testing purposes (needs docker installed)
    # False: run training job on an AWS instance
    run_locally: bool = False

    # define names for project, experiment and run
    project_name = "latent-accompaniment-generation"
    # experiment_name = "bart-drums-mixdata"

    checkpoint_ec2_path = "/opt/ml/checkpoints"
    # checkpoint_s3_path = "s3://latent-accompaniment-generation/checkpoints/ash-drums"

    # NOTE: job name has to be unique
    now = datetime.now()
    datestring = now.strftime('%Y-%m-%d-%H-%M')
    job_name = experiment_name + "-" + datestring

    # define session and role
    sagemaker_session = (sagemaker.local.LocalSession()
                         if run_locally else sagemaker.Session())
    role = ("arn:aws:iam::076456026604:role/service-role/"
            "AmazonSageMaker-ExecutionRole-20240407T195108")

    # define input and output directories
    if run_locally:
        data_path: str = "file://./data/mtg-jamendo-low/"
        output_path: str = ("file://./checkpoints/"
                            f"{project_name}/{experiment_name}/{job_name}")
    else:
        # NOTE: whatever directory is specified in here will be downloaded to the
        # training machine in its entirety (if using "File" input mode, which is
        # the default). So make sure that the s3 directory specified here only
        # contains data that is needed for the training.
        # WRONG: s3://audio-data-bucket/data/       (.../MNIST, .../moisesdb)
        # OK:    s3://audio-data-bucket/data/MNIST
        data_path: str = "s3://audio-data-bucket/data/lag-data/" + data_dir
        weights_path: str = "s3://latent-accompaniment-generation/weights/"
        output_path = f"s3://{project_name}/{experiment_name}"
        # ckp_path = checkpoint_s3_path

    # instance_type: str = "ml.g4dn.xlarge"
    # instance_type: str = "ml.m5.xlarge"
    # instance_type: str = "ml.g5.2xlarge" # 1x nvidia A10
    # instance_type: str = "ml.g5.12xlarge"  # 4x nvidia A10
    # instance_type: str = "ml.p4d.24xlarge"  # 8x nvidia A100 40gb 💀
    # instance_type: str = "ml.p4de.24xlarge"  # 8x nvidia A100 80gb 💀
    # instance_type: str = "ml.trn1.2xlarge"

    # max_training_time = 3 * 24 * 60 * 60  # 3 days
    # max_training_time = 16 * 60 * 60  # 16 hours
    max_training_time = max_training_hours * 60 * 60

    estimator = sagemaker.pytorch.PyTorch(
        # entry_point="scripts/train_bart.py",
        entry_point=entry_point,
        role=role,
        max_run=max_training_time,
        instance_count=1,
        framework_version="2.1.0",
        py_version="py310",
        dependencies=["requirements.txt"],
        source_dir="src",
        output_path=output_path,
        instance_type="local" if run_locally else instance_type,
        local_code=run_locally,
        # input_mode="FastFile",
        # input_mode="File",
        input_mode=input_mode,
        checkpoint_local_path=checkpoint_ec2_path,
        # checkpoint_s3_uri=checkpoint_s3_path,
        # distribution={"pytorchddp": {
        #     "enabled": "true"
        # }},
    )

    estimator.fit(
        {
            "data": data_path,
            # "ckp": ckp_path,
            "weights": weights_path,
        },
        job_name=job_name,
    )
