#!/usr/bin/python
# -*- coding: utf-8 -*-

# This code is based on https://github.com/toandaominh1997/EfficientDet.Pytorch/blob/master/models/retinahead.py

# Copyright 2020 toandaominh1997. All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from torch import nn, Tensor
import torch
from hydranet.models.module import Anchors, ClipBoxes, BBoxTransform
from hydranet.models.losses import FocalLoss
from torchvision.ops import nms


class BboxDetectionHead(nn.Module):
    def __init__(
        self,
        object_detection_threshold: float = 0.01,
        object_detection_iou_threshold: float = 0.5,
        is_training: bool = False,
    ) -> None:
        super().__init__()
        self.is_training = is_training
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.threshold = object_detection_threshold
        self.iou_threshold = object_detection_iou_threshold
        self.criterion = FocalLoss()

    def forward(self, inputs: Tensor) -> Tensor:
        if self.is_training:
            inputs, annotations = inputs
        else:
            inputs = inputs
        classification = torch.cat([out for out in inputs[0]], dim=1)
        regression = torch.cat([out for out in inputs[1]], dim=1)
        anchors = self.anchors(inputs)
        if self.is_training:
            return self.criterion(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, inputs)
            scores = torch.max(classification, dim=2, keepdim=True)[0]
            scores_over_thresh = (scores > self.threshold)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                print("No boxes to NMS")
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]
            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]
            anchors_nms_idx = nms(
                transformed_anchors[0, :, :],
                scores[0, :, 0],
                iou_threshold=self.iou_threshold,
            )
            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]
