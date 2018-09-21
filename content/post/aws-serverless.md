+++
date = "2018-09-16"
description = ""
tags = ["machine learning", "cloud"]
draft = false
author = "Ritchie Vink"
title = "Deploy any machine learning model serverless in AWS"
og_image = "/img/post-17-serverless_model/serverless-architecture.png"
+++

{{< figure src="/img/post-17-serverless_model/serverless-architecture.png" >}}

When a machine learning model goes into production, it is very likely to be idle most of the time. There are a lot of use cases, where a model only needs to run inference when new data is available. If we do have such a use case and we deploy a model on a server, it will eagerly be checking for new data, only to be disappointed for most of its lifetime and meanwhile you pay for the live time of the server. 

Now the cloud era has arrived, we can deploy a model serverless. Meaning we only pay for the compute we need, and spinning up the resources when we need them. In this post we'll define a serverless infrastructure for a machine learning model. This infrastructure will be hosted on AWS.

## TL;DR
Deploy a serverless model. Take a look at:

[https://github.com/ritchie46/serverless-model-aws](https://github.com/ritchie46/serverless-model-aws)

For the code used in this blog post:

[https://github.com/ritchie46/serverless-model-aws/tree/blog](https://github.com/ritchie46/serverless-model-aws/tree/blog)

## Architecture
The image below shows an overview of the serverless architecture we're deploying. For our infrastructure we'll use at least the following AWS services:

* AWS S3: Used for blob storage.
* AWS Lambda: Executing functions without any servers. These functions are triggered by various events.
* AWS SQS: Message queues for interaction between microservices and serverless applications.
* AWS ECS: Run docker containers on AWS EC2 or AWS Fargate.

<br>

{{< figure src="/img/post-17-serverless_model/architecture.svg" title="The serverless architecture" >}}

The serverless application works as follows. Every time new data is posted on a S3 bucket, it will trigger a Lambda function. This Lambda function will push some meta data (data location, output location etc.) to the SQS Queue and will check if there is already an ECS task running. If there is not, the Lambda will start a new container. The container, once started, will fetch messages from the SQS Queue and process them. Once there are no messages left, the container will shut down and the
Batch Transform Job is finished!

## Docker image

Before we'll start with the resources in the cloud, we will prepare a Docker image, in which our model will reside.

### Requirements
You'll need make sure you've got the following setup. You need to have access to an AWS account and install and configure the [aws cli](https://aws.amazon.com/cli/), and [Docker](https://docs.docker.com/install/). The aws cli enables us to interact with the AWS Cloud from the command line and Docker will help us containerize our model. 

To make sure we don't walk into permission errors in AWS, make sure you've created admin access keys and add them to _~/.aws/credentials_.

```
[default]
aws_access_key_id = <key-id> 
aws_secret_access_key = <secret-key>
```

### File structure
For creating the Docker image we will create the following file structure. On the root of our project we have a _Dockerfile_ and the Python dependencies in _requirements.txt_.

```
project
|   Dockerfile
|   requirements.txt
|   build_and_push.sh
|
|---src
|   |   batch_transform.py
|   |
|   |---cloudhelper
|   |   |   __init__.py
|   |   
|   |---model
|   |   |   __init__.py
|   |   |   transform.py

```

Let's go through the files one by one!

### Dockerfile
In the Dockerfile we start from the Python 3.6 base image. If you depend on pickled files, make sure you use the same Python and library dependencies in the Dockerfile as the one you've used to train your model!

The Dockerfile is fairly straightforward. We copy our project files in the image
```
FROM python:3.6

# first layers should be dependency installs so changes
# in code won't cause the build to start from scratch.
COPY requirements.txt /opt/program/requirements.txt

RUN pip3 install --no-cache-dir -r /opt/program/requirements.txt

# Set some environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY src /opt/program
COPY serverless/batch-transform/serverless.yml /opt/program/serverless.yml
WORKDIR /opt/program

CMD python batch_transform.py
```

### requirements.txt

The _requirements.txt_ has got some fixed requirements for accessing AWS and reading YAML files. The other dependencies can be adjusted for any specific needs for running inference with your model.

``` text
# always required
boto3
pyyaml

# dependencies for the custom model
numpy
scipy
scikit-learn
pandas
pyarrow
```
### cloudhelper/\_\_init\_\_.py
These are just some helper functions for retrieving and writing data from and to S3.

``` python
import io
import boto3


def open_s3_file(bucket, key):
    """
    Open a file from s3 and return it as a file handler.
    :param bucket: (str)
    :param key: (str)
    :return: (BytesIO buffer)
    """
    f = io.BytesIO()
    bucket = boto3.resource('s3').Bucket(bucket)
    bucket.Object(key).download_fileobj(f)
    f.seek(0)
    return f


def write_s3_file(bucket, key, f):
    """
    Write a file buffer to the given S3 location.
    :param bucket: (str)
    :param key: (str)
    :param f: (BytesIO buffer)
    """
    f.seek(0)
    bucket = boto3.resource('s3').Bucket(bucket)
    bucket.Object(key).upload_fileobj(f)


def write_s3_string(bucket, key, f):
    """
    Write a StringIO file buffer to S3.
    :param bucket: (str)
    :param key: (str)
    :param f: (StringIO buffer)
    """
    try:
        f.seek(0)
        bf = io.BytesIO()
        bf.write(f.read().encode('utf-8'))
        bf.seek(0)
        bucket = boto3.resource('s3').Bucket(bucket)
        bucket.Object(key).upload_fileobj(bf)
    except Exception as e:
        print('Exception: ', e)
    return True

```
### model/\_\_init\_\_.py

In this file we will load the settings from a YAML settings file that doesn't exist yet. Don't worry, we'll get to that in a bit.

In the _ModelWrap_ class we've got a getter method (with @property) above. With this we can access the model from S3 or from memory. And a _predict_ method. This method will be called when running inference.

``` python
from cloudhelper import open_s3_file
import pandas as pd
import os
import yaml
import pickle


class ModelWrap:
    def __init__(self):
        if os.path.exists('../serverless/batch-transform/serverless.yml'):
            p = '../serverless/batch-transform/serverless.yml'
        else:
            p = 'serverless.yml'

        with open(p) as f:
            self.config = yaml.load(f)['custom']['dockerAvailable']

        self._model = None

    @property
    def model(self):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if self._model is None:
            f = open_s3_file(self.config['BUCKET'], self.config['MODEL_PKL'])
            self._model = pickle.load(f)
        return self._model

    def predict(self, x):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        id = x.iloc[:, 0]
        x = x.iloc[:, 1:]
        p = self.model.predict_proba(x)[:, 1]
        return pd.DataFrame({'id': id, 'activation': p})


modelwrapper = ModelWrap()
```

### model/transform.py
This is the file in which the application comes together. Here we fetch messages from the SQS queue. It is assumed that the messages are in JSON format with the following key-value information:

``` json
{   bucket: <input-data-bucket>,
    key: <input-data-key>,
    output_bucket: <storage-bucket>,  # Location to write the data to
    output_key: <storage-key>
}
```
In the _BatchTransformJob.process\_q_ method we process the messages that are currently available. Finally, in the _run\_batch\_transform\_job_ function we call this method until there are no messages left.

``` python
import json
import os
import io
import time
from datetime import datetime
import boto3
import pandas as pd
from model import modelwrapper
from cloudhelper import open_s3_file, write_s3_string

sqs = boto3.resource('sqs', region_name=modelwrapper.config['AWS_REGION'])
s3 = boto3.resource('s3', region_name=modelwrapper.config['AWS_REGION'])


class BatchTransformJob:
    def __init__(self, q_name):
        self.q_name = q_name
        self.q = sqs.get_queue_by_name(
            QueueName=q_name
        )
        self.messages = None

    def fetch_messages(self):
        self.messages = self.q.receive_messages()
        return self.messages

    def process_q(self):
        for message in self.messages:
            m = json.loads(message.body)

            print(f"Downloading key: {m['key']} from bucket: {m['bucket']}")

            f = open_s3_file(m['bucket'], m['key'])
            x = pd.read_csv(f)

            print('Invoked with {} records'.format(x.shape[0]))
            # Do the prediction
            predictions = modelwrapper.predict(x)

            f = io.StringIO()
            predictions.to_csv(f, index=False)
            if write_s3_string(bucket=m['output_bucket'],
                               key=os.path.join(f"{m['output_key']}",
                                                datetime.now().strftime('%d-%m-%Y'), f"{int(time.time())}.csv"),
                               f=f):
                print('Success, delete message.')
                message.delete()


def run_batch_transform_job():
    btj = BatchTransformJob(os.environ['SQS_QUEUE'])

    t0 = time.time()
    btj.fetch_messages()

    c = 0
    while len(btj.messages) == 0:
        c += 1
        back_off = 2 ** c
        print(f'No messages ready, back off for {back_off} seconds.')

        time.sleep(back_off)  # back off
        btj.fetch_messages()

        if (time.time() - t0) > 900:
            print('Maximum time exceeded, close the container')
            return False

    print('Found messages, process the queue')
    btj.process_q()

```
### build\_and\_push.sh

This is just a convenience file for building the docker image and pushing it to AWS ECR (Elastic Container Registry). 

``` bash
#!/usr/bin/env bash

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi


# Get the region defined in the current configuration (default to us-west-2 if none defined)
# us-west-1 == ireland
region=$(aws configure get region)
region=${region:-us-west-1}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build  -t ${image} .
docker tag ${image} ${fullname}

docker push ${fullname}
```

## Serverless framework
We are not going to build and push our Docker image just jet, as a lot of the code in the image requires some secret YAML file we have not defined.

This YAML file is actually the template for the [serverless platform](https://serverless.com/). Because it is easy to have all the settings in one file, we also add the settings for our Docker image in this template.

The serverless framework lets us define a whole serverless AWS Cloudformation stack. This means we can define our infrastructure as code. This of course has a lot of benefits like source control, parameterized deployment etc.

### Requirements
First install [serverless](https://serverless.com/framework/docs/getting-started/). You'll need [NPM and Node.js](https://www.npmjs.com/get-npm) for this.

`$ npm install -g serverless`

For this to work I've also assumed that 3 things are already defined in AWS. These things are:

* An AWS IAM Role called _ecsTaskExecutionRole_ with Full SQS, S3 and ECS access. 
* An AWS subnet
* An AWS security group

You can create these in the AWS Cloud console or with the aws cli. For the _ecsTaskExecutionRole_ the ARN [(Amazone Resource Name)](https://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html) should be added to the template. For the subnet and the security group, the name you have given to the resources.

These 3 resources can also be defined in the serverless template. I've chosen for this solution as I wanted to deploy multiple models and share the VPC (Virtual Private Cloud) settings.

### Model
This whole architecture is of course useless without a machine learning model. I have gone through a huge effort to develop a tremendous model for a very hard challenge. The fruits of my labor can be downloaded by clicking the following links:

* [iris-dataset.csv](/download/iris.csv)

* [model.pkl](/download/model.pkl)

The model file needs to be uploaded to a S3 location by your liking.
For now I assume they are located at:

`s3://my-bucket/model.pkl`

### File structure
The file structure we have defined earlier will be updated to include the code for the serverless infrastructure and the actual AWS Lambda function.

```
project
|   Dockerfile
|   requirements.txt
|   build_and_push.sh
|
|---src
|   |   ...
|   
|---serverless
|   |  
|   |---batch-transform
|   |   |   serverless.yml
|   |   |   handler.py   
```

### serverless/batch-transform/handler.py
This file contains the actual AWS Lambda function we are deploying. 
This Lambda is triggered by a 'new data' trigger. In the _lambda\_handler_ function we push the bucket and key information of this new data to an SQS queue. Next we check if there are any ECS tasks running. If there aren't we start our container to process the messages on the SQS queue.

Note that there are a lot of environment variables used (os.environ). Those environment variables are set in the serverless template.

``` python
import json
import os
from datetime import date, datetime
import boto3


# SETTINGS
DESIRED_COUNT = int(os.environ['DESIRED_COUNT'])
OUTPUT_BUCKET = os.environ['OUTPUT_BUCKET']
OUTPUT_KEY = os.environ['OUTPUT_KEY']
SQS_NAME = os.environ['RESOURCE_NAME']
ECS_CLUSTER = os.environ['RESOURCE_NAME']
TASK_DEFINITION = os.environ['RESOURCE_NAME']
SUBNET = os.environ['SUBNET']
SECURITY_GROUP = os.environ['SECURITY_GROUP']

s3 = boto3.resource('s3')
sqs = boto3.resource('sqs')
ecs_client = boto3.client('ecs')


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


def lambda_handler(event, context):

    event_info = event['Records'][0]['s3']

    q = sqs.get_queue_by_name(QueueName=SQS_NAME)
    message = json.dumps(dict(
        bucket=event_info['bucket']['name'],
        key=event_info['object']['key'],
        output_bucket=OUTPUT_BUCKET,
        output_key=OUTPUT_KEY))

    print(f'Add {message} to queue')
    response = q.send_message(
        MessageBody=message
    )

    if len(ecs_client.list_services(cluster=ECS_CLUSTER)['serviceArns']) == 0:

        print('RUN ECS task')

        response = ecs_client.run_task(
            cluster=ECS_CLUSTER,
            taskDefinition=TASK_DEFINITION,
            count=DESIRED_COUNT,
            launchType='FARGATE',
            networkConfiguration=dict(
                awsvpcConfiguration=dict(subnets=[SUBNET],
                                         securityGroups=[SECURITY_GROUP],
                                         assignPublicIp='ENABLED')

            ),
        )
    else:
        print('ECS cluster already running, lay back')

    return {
        "statusCode": 200,
        "body": json.dumps(response, default=json_serial)
    }
```

### serverless/batch-transform/serverless.yml
This file is where all the settings for both the Docker image and the serverless infrastructure are defined. I won't explain the serverless syntax here, but I do encourage you to take a look at the serverless documentation to get a grasp of what is going on. The [quick start](https://serverless.com/framework/docs/providers/aws/guide/quick-start/) is a good place to begin.

Below the whole file is shown. Everything defined under the _custom_ keyword, are the settings that need to be changed for your specific model. The other keywords are the specification of the serverless architecture. Everything enclosed in `${self:<some-variable>}` are variables.

A quick overview of what is defined in this file:

* `fillSQS`: Here the AWS Lamba is defined.
* `ContainerLogs`: So we are able to view the logs in the container once deployed.
* `NewDataQueue`: Defines the SQS queue that is needed.
* `ECSCluster`: The container task will run in an ECS Cluster.
* `BatchTransformTask`: The task definition for the container.

The serverless YAML file also requires an URI (under _custom.image_) to the Docker image. To get this, we need to push the Docker image to ECR and retrieve the URI.

`$ ./build_and_push.sh <image-tag>`

`$ aws ecr describe-repositories | grep repositoryUri`

``` yaml 

service: ${self:custom.prefix}-${self:custom.usecase}-${self:custom.model} # NOTE: update this with your service name

# Custom are the variables for this template.
custom:
  dockerAvailable:
    # These setting will be exposed for the model in the docker image
    BUCKET: my-bucket
    MODEL_PKL: model.pkl
    AWS_REGION: eu-west-1

  # Docker image that will be deployed
  image: <repository url>
  desiredTaskCount: 1

  # Settings for the naming of new AWS resources
  prefix: <str> Resources made in AWS will have this prefix
  usecase: <str> Resoures made in AWS will have this name
  model: <str> Name of the model. Name will be given to the Resources in AWS

  # Bucket & key to where the results are written
  outputBucket: <bucket>
  outputKey: <key>

  # Bucket that will be generated for this stack. New data should be deployed here.
  bucket: ${self:custom.prefix}-${self:custom.usecase}-${self:custom.model}-new-data

  # File type that should trigger the Lambda
  triggerExtension: .csv

  # Subnet and security group names in which the AWS Task should run.
  subnet:  <subnet name>
  securityGroup: <security group name>

  # ARN of the Role that will be assigned to the Task. It needs SQS, S3 and ECS access
  ecsTaskExecutionRole: <arn of role with the needed permissions>

## Setting can be changed, but this is not required.
provider:
  name: aws
  runtime: python3.6
  stage: dev
  region: ${self:custom.flask.AWS_REGION}
  iamRoleStatements:
    - Effect: Allow
      Action:
        - s3:*
        - sqs:*
        - ecs:*
        - iam:PassRole
      Resource: "*"
  environment:
      RESOURCE_NAME: ${self:custom.prefix}-${self:custom.usecase}-${self:custom.model}
      OUTPUT_BUCKET: ${self:custom.outputBucket}
      OUTPUT_KEY: ${self:custom.outputKey}
      SUBNET: ${self:custom.subnet}
      SECURITY_GROUP: ${self:custom.securityGroup}
      DESIRED_COUNT: ${self:custom.desiredTaskCount}


functions:
  fillSQS:
    handler: handler.lambda_handler
    name: ${self:custom.prefix}-${self:custom.usecase}-${self:custom.model}
    events:
      - s3:
          bucket: ${self:custom.bucket}
          event: s3:ObjectCreated:*
          rules:
            - suffix: ${self:custom.triggerExtension}

resources:
  Resources:
    # Needed for the container logs
    ContainerLogs:
      Type: AWS::Logs::LogGroup
      Properties:
        LogGroupName: /ecs/${self:custom.prefix}-${self:custom.usecase}-${self:custom.model}

    NewDataQueue:
      Type: AWS::SQS::Queue
      Properties:
        QueueName: ${self:custom.prefix}-${self:custom.usecase}-${self:custom.model}

    ECSCluster:
      Type: AWS::ECS::Cluster
      Properties:
        ClusterName: ${self:custom.prefix}-${self:custom.usecase}-${self:custom.model}

    BatchTransformTask:
      Type: AWS::ECS::TaskDefinition
      Properties:
        TaskRoleArn: ${self:custom.ecsTaskExecutionRole}
        ExecutionRoleArn: ${self:custom.ecsTaskExecutionRole}
        Cpu: 2048
        Memory: 16384
        Family: ${self:custom.prefix}-${self:custom.usecase}-${self:custom.model}
        NetworkMode: awsvpc
        RequiresCompatibilities:
          - FARGATE
        ContainerDefinitions:
          -
            Name: ${self:custom.prefix}-${self:custom.usecase}-${self:custom.model}
            Image: ${self:custom.image}
            Environment:
              -
                Name: SQS_QUEUE
                Value: ${self:custom.prefix}-${self:custom.usecase}-${self:custom.model}
            Command:
              - python
              - batch_transform.py
            LogConfiguration:
              LogDriver: awslogs
              Options:
                awslogs-region: ${self:provider.region}
                awslogs-group: ${self:resources.Resources.ContainerLogs.Properties.LogGroupName}
                awslogs-stream-prefix: ecs
```

## Deployment
Now the Docker image and the code for the serverless infrastructure is defined, deployment with the serverless cli is easy!

`$ cd serverless/batch-transform && serverless deploy`

This command deploys our complete infrastructure as a Cloudformation stack. Take a look at the Cloudformation page in the AWS Cloud Console to see which AWS Resources are created. 

To see the magic in action, drop the iris.csv file in the input-bucket you have defined!

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
  });
  </script>

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<head>

<style>

.formula-wrap {
overflow-x: scroll;
}

</style>

</head>
