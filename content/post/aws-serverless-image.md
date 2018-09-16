+++
date = "2018-09-13"
description = ""
tags = ["machine learning", "cloud"]
draft = true
author = "Ritchie Vink"
title = "Serverless model deployment in AWS"
og_image = "/img/post-17-serverless_model/serverless-architecture.png"
+++

When a machine learning model goes into production, it is very likely to be idle most of the time. There are a lot of use cases, where a model only needs to run inference when new data is available. If we do have such a use case and we deploy a model on a server, it will eagerly be checking for new data, only te be dissapointed for most of its lifetime and meanwhile you pay for the live time of the server. 

Now the cloud era has arrived, we can deploy a model serverless. Meaning we only pay for the compute we need, and spinning up the resources when we need them. In this post we'll define a serverless infrastructure for a machine learning model. This infrastructure will be hosted on AWS.

## Architecture
The image below shows an overview of the serverless architecture we're deploying. For our infrastructure we'll use at least the following AWS services:

* AWS S3: Used for blob storage.
* AWS Lambda: Executing functions without any servers. These functions are triggered by various events.
* AWS SQS: Message queues for interaction between microservices and serverless aplications.
* AWS ECS: Run docker containers on AWS EC2 or AWS Fargate.

<br>

{{< figure src="/img/post-17-serverless_model/architecture.svg" title="The serverless architecture" >}}

The serverless application works as follows. Every time new data is posted on a S3 bucket, it will trigger a Lambda function. This Lambda function will push some meta data (data location, output location etc.) to the SQS Queue and will check if there is already a ECS task running. If there is not, the Lambda will start a new container. The container, once started, will fetch messages from the SQS Queue and process them. Once there are no messages left, the container will shut down and the
Batch Transform Job is finished!

## Docker image

Before we'll start with the resources in the cloud, we will prepare a Docker image, in which our model will reside.

We have gone through a huge effort to develop a tremendous model for a very hard challenge. The fruits of our labor can be downloaded by clicking the following links:

* [iris-dataset.csv](/download/iris.csv)

* [model.pkl](/download/model.pkl)

Furthermore you'll need make sure you've got the following setup. You need to have access to a AWS account and install the [aws cli](https://aws.amazon.com/cli/), and [Docker](https://docs.docker.com/install/). The aws cli enables us to interact with the AWS Cloud from the command line and Docker will help us containerize our model. 





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
