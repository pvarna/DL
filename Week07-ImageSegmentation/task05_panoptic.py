import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import functional
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.optim as optim
import torchmetrics


def main():
    semantic_model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)

    semantic_model.classifier = FCNHead(2048, 201)
    semantic_model.aux_classifier = None

    instance_model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = instance_model.roi_heads.box_predictor.cls_score.in_features
    instance_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 81)
    in_features_mask = instance_model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    instance_model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, 81)

    semantic_model.load_state_dict(torch.load('best_fcn_resnet50.pt'))
    instance_model.load_state_dict(torch.load('best_maskrcnn_resnet50_v2.pt'))

    semantic_model.eval()
    instance_model.eval()

    image_path = "../DATA/relabelled_coco/images/000000001000.jpg"
    image_pil = Image.open(image_path)

    image_tensor = functional.to_tensor(image_pil).unsqueeze(0)

    with torch.no_grad():
        semantic_masks = semantic_model(image_tensor)["out"]
        print(semantic_masks.shape)

    semantic_mask = torch.argmax(semantic_masks, dim=1)

    with torch.no_grad():
        instance_masks = instance_model(image_tensor)[0]['masks']
        print(instance_masks.shape)

    panoptic_mask = torch.clone(semantic_mask)
    instance_id = 201

    for mask in instance_masks:
        panoptic_mask[mask > 0.5] = instance_id
        instance_id += 1

    plt.imshow(panoptic_mask.squeeze(0), cmap='nipy_spectral')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
