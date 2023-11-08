import cv2
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
torch.cuda.empty_cache()
import numpy as np
import matplotlib.pyplot as plt

def sam_segmentation(path_weigths):
  # weigths of pretrained sam with medical iamge https://huggingface.co/flaviagiammarino/medsam-vit-base
  sam = sam_model_registry['vit_b'](checkpoint=path_weigths) #adaptation on dropbox
  mask_generator = SamAutomaticMaskGenerator(sam)
  DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  sam.to(device=DEVICE)
  return mask_generator

def sam_inference(img,mask_generator):
  final_mask = 0
  masks = mask_generator.generate(img)

  for mask in masks:
    mask_ = np.array(mask['segmentation'], dtype=np.uint8)
    
    if np.unique(mask_).shape[0] == 2:
      final_mask = final_mask + mask_

  final_mask = np.where(final_mask > 1, 1, final_mask)
  
  return final_mask