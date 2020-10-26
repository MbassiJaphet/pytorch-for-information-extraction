---
id: conclusion
sidebar_label: Conclusion
title: Tutorial Conclusion
slug: /conclusion/
---

Wow, we are almost the end of this tutorial !

## Final words

This tutorial introduced the basics of object detection, instance segmentation, and object classification using Pytorch. It even went as far as demonstrating how it is possible to use Pytorch features to perform information extraction. It presented a methodology for information extraction which we implemented using Pytorch. In the course of implementing the different modules of this methodology, we leveraged a lot of Pytorch features by doing more or less of the following:

*   Create custom dataset classes inheriting [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset).
*   Initialize datasets for training, validation, and testing.
*   Explore and visualize datasets.
*   Define data loaders for our datasets.
*   Load and fine-tune pre-trained models i.e. Mask R-CNN showcased in this tutorial.
*   Train Mask R-CNN to carry out object detection and instance segmentation.
*   Interpret and visualize Mask R-CNN model predictions.
*   Load datasets using Pytorch in-built [torchvision.datasets.ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.ImageFolder).
*   Build a simple image classifier.
*   Define and apply transformations on training, validation, and testing datasets.
*   Select criterion, define optimizer and training loop for image classifier.
*   Compute prediction of image classifier on sample data.
*   Save model and optimizer state dictionaries as checkpoints.
*   Load model and optimizer from checkpoints.
*   Resume training using checkpoints.

## What comes next
*   Make the detection module more robust to truncated documents.
*   Develop and train a custom **OCR engine** using Pytorch.