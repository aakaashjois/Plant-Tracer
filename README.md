# Plant Tracer
A deep learning approach to tracking the apex of a moving plant.

## Introduction
Plant Tracer is an app designed to enable analysis of plant movement from time-lapse videos. This repository contains a small part of the complete project. Here, I try to track the movement of the plant using a deep learning model to make it robust and work on videos with occlusion while tracking plants.

## Architecture
The architecture follows the architecture in [GOTURN](https://arxiv.org/pdf/1604.01802.pdf). The `CaffeNet` pretrained on `CIFAR-100` has been replaced with `AlexNet` pretrained on `ImageNet`.  The architecture of the model is shown below.
![](./misc/architecture.png?raw=True "Architecture")

## Instructions
1. Obtain data from [Plant Tracer homepage](https://www.planttracer.com/).
2. Clone this repository.
3. This project uses `conda` environment. Create the conda virtual environment using `conda env create -f environment.yaml`.
4. Modify the `run.py` file and run it to start the training procedure. This project uses [Comet](https://www.comet.ml) for all visualizations. Add your comet API key in the `run.py` file to see visualizations.
5. The models are saved in `models` folder and validation and testing results are stored in the `logs` directory and can be visualized with the `viz.py` file.

## Report
This repository contains the code used for submission of this [Report](./misc/Report.pdf).

## Result
The output from validation and testing of the model can be seen below.
#### Validation
![](./misc/validation.gif?raw=True "Validation")
#### Tracking Test
![Testing](./misc/testing.gif?raw=True "Testing")

## License
Plant-Tracer is released under [Apache License 2.0](./LICENSE.md).

## Author
This repository and the approach has been created by [Aakaash Jois](https://aakaashjois.com).

The complete Plant Tracer application has multiple authors and details related to that can be found on [Plant Tracer homepage](https://www.planttracer.com).
