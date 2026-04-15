# HawkAI
The complete intelligent systems system for Computer Vision tasks like object detection and mapping. 
It integrates with the Intelligent Systems Ground System (GS) and Imaging System's Ground Server for real time communication.
In addition to its light-weight process for mission-critical ML tasks, it also hosts some other important
functionality, including a script to transform annotated data 
into an easily-processable CSV for ML training.

## Docker

You'll need to have docker installed on your machine. You can download it from [here](https://www.docker.com/products/docker-desktop/).

### Run everything in docker at once with docker-compose

```bash
docker-compose up --build
```

### Otherwise take the following steps:

1. Build Docker image

```bash
docker build -t intsys-client .
```

2. Run Docker container with GS server at 192.168.1.2:9000

```bash
docker run --rm -it --name intsys-client intsys-client
```

Run Docker container locally or with custom IP address

Locally:
```bash
docker run --rm -it --name intsys-client intsys-client --ip host.docker.internal:9000
```

Custom IP address:
```bash
docker run --rm -it --name intsys-client intsys-client --ip <ip_address>
```

### To Generate Map, use the Intsys GS

### Download model weights

Download the model weights from the Box folder and place them in the `model_weights` directory. The required weight files are:

- `model_weights/maskrcnn_Mar52025.pth`
- `model_weights/filter_classifier_weights.pt`
- `model_weights/number_classifier_weights_resnet18.pt`

# Other Utilities
Other functionality exists in`./utils`. The only tool currently is a script to process annotated data from CVAT
into a `.csv` for further processing / ML training. To use it, check out the `README.md` in that sub-directory.
