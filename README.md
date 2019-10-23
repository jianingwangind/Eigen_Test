# [Scaling Object Detection by Transferring Classification Weights](https://arxiv.org/pdf/1909.06804.pdf):

## Problem Setting
* To transfer knowledge from source task(classification) to target task(detection). By knowledge, the weights coming from the final fully connected layers are referred.
* Annotated detection dataset and unlabeled classification dataset are used to train the whole network.
* Once trained, the target task network(detector) can predict well on those during-training-unseen categories.

## Basic Structure
* Weight transfer network is implemented as an Auto-encoder, which maps the classification weights to the detection weights.
* The detection weights coming from the last step are directly used to accomplish the target task. Namely, the fc layers from classification network, the weight transfer network and the fc layers from the detector are in serial, which allows the gradients flow from the detector to the weight transfer network.

## Main Contributions
* Standard normalization applied to the classification weights, because of the unbalanced class distribution. Then different per-class weight vector will contribute comparably to the prediction of detection weights.
* Auto-encoder structure along with a reconstruction loss allows better to preserve the semantic information, while trying to learn the most discriminative class information.
