import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
import random
from scipy import ndimage
import copy
import cv2
from datasets import Dataset
from PIL import Image
from torch.utils.data import Dataset as DatasetTorch


from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
from torch.optim import RMSprop
import monai
from transformers import SamModel
from torch.utils.tensorboard import SummaryWriter
from torch import nn

from sklearn.metrics import accuracy_score

import csv

from transformers import SamProcessor
from torch.utils.data import DataLoader

from statistics import mean


def datasetToSAM(X_train,y_train):
  """X_train: numpy array with training images dimentions (batch,w,h,c)
     y_train: Mask of X_train iamge dimentions (batch,w,h,1)
     train_dataloader: dataset with specific format to train SAM"""

  dataset_dict = {
      "image": [Image.fromarray(img) for img in X_train],
      "label": [Image.fromarray(mask) for mask in y_train],}

  #Create dict and organizate to "Dataset" format
  dataset = Dataset.from_dict(dataset_dict)

  #Create an instance of the SAMDataset
  processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
  train_dataset = SAMDataset(dataset=dataset, processor=processor)
  train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True,drop_last=False)

  return train_dataloader,dataset,processor,train_dataset


def getBundingBox(ground_truth_map):
  """Extract bounding box of ground truth mask, its necesary
     to train SAM
     ground_truth_map: Mask format (w,h)
     Note: Its possible improve create bbox for obj"""
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox


class SAMDataset(DatasetTorch):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = np.array(item["label"])

    # get bounding box prompt
    prompt = getBundingBox(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs

def ParameterFineTunningSam(plot_model):
  # Load the model
  model = SamModel.from_pretrained("facebook/sam-vit-base")
  model.load_state_dict(torch.load('./TRAIN/SAM/weigths/vit_DSB.pth'))

  # make sure we only compute gradients for mask decode
  for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
      param.requires_grad_(False)

  optimizer = RMSprop(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
  seg_loss = monai.losses.FocalLoss(reduction='mean')
  #automatization optimizer an loss
  if plot_model:
    print(model)

  return model, optimizer, seg_loss

def DICE_metric(y_true, y_pred):

  y_true = tf.cast(y_true , dtype=tf.double)

  y_pred = tf.cast(y_pred  , dtype=tf.double)

  intersection = tf.reduce_sum(y_true * y_pred)
  union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
  dice = (2.0 * intersection + 1e-5) / (union + 1e-5)

  return dice

def save_besth_weigth(save_besth_accuracy,epoch_accuracy,model,path_save_weights):
  if epoch_accuracy > max(save_besth_accuracy):
    torch.save(model.state_dict(), path_save_weights)

def tensorboard_log(epoch,epoch_accuracy,val_accuracy,epoch_loss,val_loss,writer):

  writer.add_scalar("epoch_accuracy", epoch_accuracy, epoch)
  writer.add_scalar("epoch_loss", epoch_loss, epoch)
  writer.add_scalar("epoch_val_accuracy", val_accuracy, epoch)
  writer.add_scalar("epoch_val_loss", val_loss, epoch)

def csv_save_and_print(epoch,epoch_loss,val_loss,epoch_accuracy,val_accuracy,epoch_IOU,val_IOU,epoch_DICE,val_DICE,csv_filename):

  data = [
      ("Dataset","Epoch", "Loss", "Accuracy","IOU","DICE"),
      ("Train",epoch ,"{:.6f}".format(epoch_loss), "{:.6f}".format(epoch_accuracy),"{:.6f}".format(epoch_IOU),"{:.6f}".format(epoch_DICE) ),
      ("Valid",epoch ,"{:.6f}".format(val_loss), "{:.6f}".format(val_accuracy),"{:.6f}".format(val_IOU),"{:.6f}".format(val_DICE)),]
            
  print('\n')
  print(print_table(data))
  print('\n')

  with open(csv_filename , "a", newline="") as file_csv:
    writer_csv = csv.writer(file_csv)
    writer_csv.writerow([epoch, 
                         epoch_DICE, 
                         epoch_accuracy,
                         epoch_IOU,
                         epoch_loss,
                         val_DICE,
                         val_accuracy,
                         val_IOU,
                         val_loss,])

def print_table(data):

    column_lengths = [max(len(str(item)) for item in column) for column in zip(*data)]


    header = "|".join(f"{item:<{length}}" for item, length in zip(data[0], column_lengths))
    separator = "+".join("-" * length for length in column_lengths)
    print(separator)
    print(header)
    print(separator)


    for row in data[1:]:
        row_str = "|".join(f"{item:<{length}}" for item, length in zip(row, column_lengths))
        print(row_str)
        print(separator)


####################################################################################################
#Validation

def validation_model(device,model,val_dataloader,seg_loss):

  model.eval()

  val_accuracy_metric = tf.keras.metrics.Accuracy()
  val_IOU_metric = tf.keras.metrics.IoU(num_classes=2,target_class_ids=[0])

  val_accuracy_metric.reset_states()
  val_IOU_metric.reset_states()
  val_steps_losses = []
  val_steps_DICE = []

  with torch.no_grad():
    for val_batch in val_dataloader:

      outputs = model(pixel_values=val_batch["pixel_values"].to(device),
                      input_boxes=val_batch["input_boxes"].to(device),
                      multimask_output=False)
      
      
      predicted_masks = outputs.pred_masks.squeeze(1)
      predicted_masks = nn.functional.interpolate(predicted_masks,size=(128, 128),mode='bilinear',align_corners=False)
      ground_truth_masks = val_batch["ground_truth_mask"].float().to(device)
      loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
      
      predict_to_ac = predicted_masks.permute(0, 2, 3, 1).cpu().detach().numpy()
      ground_to_ac = ground_truth_masks.unsqueeze(1).permute(0, 2, 3, 1).cpu().numpy().astype(float)

      for i in range(ground_to_ac.shape[0]):
        y_truh = ground_to_ac[i,:,:,0]
        y_pred = predict_to_ac[i,:,:,0]
        y_pred = tf.clip_by_value(y_pred, 0, 1)

        val_accuracy_metric.update_state(y_truh, y_pred)
        val_IOU_metric.update_state(y_truh, y_pred)
        val_steps_DICE.append(DICE_metric(y_truh, y_pred).numpy())

      val_steps_losses.append(loss.item())

    epoch_accuracy = val_accuracy_metric.result().numpy()
    epoch_IOU = val_IOU_metric.result().numpy()
    epoch_loss = mean(val_steps_losses)
    epoch_DICE = mean(val_steps_DICE)
    print('val_accura',epoch_accuracy )

    return epoch_loss, epoch_accuracy, epoch_IOU, epoch_DICE


####################################################################################################
#Train
def SamTrain(num_epochs,model,train_dataloader,seg_loss,optimizer,val_dataloader,csv_filename,ts_log,path_save_weights):

  # with open(csv_filename , "w", newline="") as file_csv:
  #   writer_csv = csv.writer(file_csv)
  #   writer_csv.writerow(['epoch','DICE_metric','accuracy','io_u','loss','val_DICE_metric','val_accuracy','val_io_u','val_loss'])

  writer = SummaryWriter(ts_log)


  device = "cuda" if torch.cuda.is_available() else "cpu"

  model.to(device)

  save_besth_accuracy = [0]

  for epoch in range(123,300):

      model.train()
      steps_losses = []
      steps_DICE = []

      accuracy_metric = tf.keras.metrics.Accuracy()
      IOU_metric = tf.keras.metrics.IoU(num_classes=2,target_class_ids=[0])

      accuracy_metric.reset_states()
      IOU_metric.reset_states()

      for batch in tqdm(train_dataloader):

        # forward pass
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)

        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        predicted_masks = nn.functional.interpolate(predicted_masks,size=(128, 128),mode='bilinear',align_corners=False)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

   
        predict_to_ac = predicted_masks.permute(0, 2, 3, 1).cpu().detach().numpy()
        ground_to_ac = ground_truth_masks.unsqueeze(1).permute(0, 2, 3, 1).cpu().numpy().astype(float)

        #Calculate metrics 
        for i in range(ground_to_ac.shape[0]):
          y_truh = ground_to_ac[i,:,:,0]
          y_pred = predict_to_ac[i,:,:,0]

          y_pred = tf.clip_by_value(y_pred, 0, 1)

          accuracy_metric.update_state(y_truh, y_pred)
          IOU_metric.update_state(y_truh, y_pred)
          steps_DICE.append(DICE_metric(y_truh, y_pred).numpy())

        steps_losses.append(loss.item())

        # backward pass (compute gradients of parameters w.r.t. loss)

        optimizer.zero_grad()
        loss.backward()

        # optimize
        optimizer.step()


      epoch_accuracy = accuracy_metric.result().numpy()
      epoch_IOU = IOU_metric.result().numpy()
      epoch_loss = mean(steps_losses)
      epoch_DICE = mean(steps_DICE)

      save_besth_weigth(save_besth_accuracy,epoch_accuracy,model,path_save_weights)
      save_besth_accuracy.append(epoch_accuracy)


      val_loss, val_accuracy, val_IOU, val_DICE = validation_model(device,model,val_dataloader,seg_loss)

      csv_save_and_print(epoch,epoch_loss,val_loss,epoch_accuracy,val_accuracy,epoch_IOU,val_IOU,epoch_DICE,val_DICE,csv_filename)

      tensorboard_log(epoch,epoch_accuracy,val_accuracy,epoch_loss,val_loss,writer)

  writer.close()




  ################################################################################
  #inference

from transformers import SamModel, SamConfig, SamProcessor
from datasets import Dataset
from PIL import Image


# Create the dataset using the datasets.Dataset class

def sam_inference(weigths_path,imgs,mask_truth):

  dataset_dict = {
      "image": [Image.fromarray(img) for img in imgs],
      "label": [Image.fromarray(mask) for mask in mask_truth],}

  dataset = Dataset.from_dict(dataset_dict)
  idx=5
  

  device = 'cuda'
  model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
  processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
  
  sam_model = SamModel(config=model_config)
  sam_model.load_state_dict(torch.load(weigths_path))


  prompt = getBundingBox(np.array(dataset[idx]["label"]))
  
  inputs = processor(dataset[idx]["image"], input_boxes=[[prompt]], return_tensors="pt")
  
  inputs = {k: v.to(device) for k, v in inputs.items()}

  sam_model.to(device)
  
  sam_model.eval()

  y_pred = []
  
  
  with torch.no_grad():
    for i in range(imgs.shape[0]):
      outputs = sam_model(**inputs, multimask_output=False)
      medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
      medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
      medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
      medsam_seg = cv2.resize(medsam_seg, (128, 128))
      y_pred.append(medsam_seg)



  return y_pred

