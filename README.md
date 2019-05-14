# Plant Tracer

A deep learning approach to tracking the apex of a moving plant.

## Introduction
Plant Tracer is an app designed to enable analysis of plant movment from time-lapse videos. This repository contains a small part of the complete project. Here, I try to track the movement of the plant using a deep learning model to make it robust and work on videos with occlusion while tracking plants.

## Architecture
The architecture follows the architecture in [GOTURN](https://arxiv.org/pdf/1604.01802.pdf). The `CaffeNet` pretrained on `CIFAR-100` has been replaced with `AlexNet` pretrained on `ImageNet`.  The architecture of the model is shown below.
![](./figures/architecture.png?raw=True "Architecture")

## Report
This repository contains the code used for submission of this [Report](./Report.pdf).

## License
Plant-Tracer is released under [Apache License 2.0](./LICENSE.md).

## Author
This repository and the approach has been created by [Aakaash Jois](https://aakaashjois.com).

The complete Plant Tracer application has multiple authors and details related to that can be found on [Plant Tracer homepage](https://www.planttracer.com).
