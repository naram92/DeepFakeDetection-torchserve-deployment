# DeepFakeDetection-torchserve-deployment

In this repository, I present the process of deploying a [deepfake detection](https://github.com/naram92/DeepFakeDetection) model via the use of [TorchServe](https://pytorch.org/serve/).

## Environment

**Code:** Install requirements
```shell
pip install -r requirements.txt
```

## Save the model locally

Once the model is trained ([notebook](https://github.com/naram92/DeepFakeDetection) or [end-to-end pipeline](https://github.com/naram92/DeepFakeDetection-mlops-model-building)), run this code and copy the model in the created directory `./model/efficientnet-b4-deepfake`

**Code:** Writing setup_config.json for deepfake model
```python
import os
import json

model_name = 'efficientnet-b4'
model_dir_path = os.path.join('./model/', model_name + '-deepfake')
os.mkdir(model_dir_path)

with open(os.path.join(model_dir_path, 'index_to_name.json'), 'w') as file:
    json.dump({'0': 'original', '1': 'fake'}, file)

with open(os.path.join(model_dir_path, 'setup_config.json'), 'w') as file:
    json.dump({
        'model_name': model_name,
        'num_classes': 1
    }, file)
```

## Build torch-model-archiver

**Create the archive of the model**  
You can check ```torch-model-archiver --help``` to learn more about its parameters. Create the directory `model_store` before running the code.

**Code:** Create the torch-model-archive of deepfake detection model based on efficientnet-b4
```shell
torch-model-archiver -f --model-name "efficientnet-b4-deepfake" --export-path ./model_store --version 1.0 \
--serialized-file ./model/efficientnet-b4-deepfake/model.pth --model-file ./model.py --handler ./handler.py \
--extra-files "./model/efficientnet-b4-deepfake/setup_config.json,./model/efficientnet-b4-deepfake/index_to_name.json"
```

## Run the model on torch-server
To be able to execute the following codes, you need to have [docker](https://docs.docker.com/get-docker/) on your machine.
By default, torch-server deploys all models in the model_store directory.

1. **If you want to run torch-server on CPU**

**Code:** Build the docker image
```shell
docker build -t torchserve-efficientnet .
```

**Code:** Start torch-server
```shell
docker run --rm -it --volume $PWD/model_store:/home/model-server/model-store -p8080:8080 -p8081:8081 \
torchserve-efficientnet
```

2. **If you want to run torch-server on GPU**  

**Code:** Build the docker image
```shell
docker build -t torchserve-efficientnet -f Dockerfile_gpu .
```

**Code:** Start torch-server
```shell
docker run --rm -it --gpus all --runtime=nvidia --volume $PWD/model_store:/home/model-server/model-store -p8080:8080 \
-p8081:8081 torchserve-efficientnet
```

**Code:** Start torch-server with a [custom configuration](https://pytorch.org/serve/configuration.html)
```shell
docker run --rm -it --volume $PWD/model_store:/home/model-server/model-store \
--mount type=bind,source=$PWD/config.properties,target=/home/model-server/config.properties -p8080:8080 -p8081:8081 \
torchserve-efficientnet
```

## Test

**Code:** To check the status of models
```shell
curl --location --request GET 'http://127.0.0.1:8081/models'
```

**Code:** Request on efficientnet-b4-deepfake - we pass it an original face image
```shell
curl http://127.0.0.1:8080/predictions/efficientnet-b4-deepfake -T ./examples/original.png
```

**Code:** Example 3 - Request on efficientnet-b4-deepfake - we pass it a fake face image
```shell
curl http://127.0.0.1:8080/predictions/efficientnet-b4-deepfake -T ./examples/fake.png
```
