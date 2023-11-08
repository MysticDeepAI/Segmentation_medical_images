import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from keras.layers import Input, Conv2D, MaxPooling2D, MaxPool2D, Conv2DTranspose, concatenate, BatchNormalization, Activation,Add, Multiply,UpSampling2D
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from tensorflow.keras.layers import ELU, LeakyReLU, ReLU, Dropout,BatchNormalization
import tensorflow as tf


actv_funtion = 'relu'

def CEM(Input,K):

  branch_1 = Conv2D(32, (K[0], 1), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(Input)
  branch_1 = Dropout(0.1)(branch_1)
  branch_1 = BatchNormalization()(branch_1)

  branch_2 = Conv2D(32, (K[1], 1), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(Input)
  branch_2 = Conv2D(32, (1,K[1]), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(branch_2)
  branch_2 = Dropout(0.1)(branch_2)
  branch_2 = BatchNormalization()(branch_2)

  branch_3 = Conv2D(32, (K[2], 1), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(Input)
  branch_3 = Conv2D(32, (1,K[2]), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(branch_3)
  branch_3 = Dropout(0.1)(branch_3)
  branch_3 = BatchNormalization()(branch_3)

  branch_4 = Conv2D(32, (K[3], 1), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(Input)
  branch_4 = Conv2D(32, (1,K[3]), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(branch_4)
  branch_4 = Dropout(0.1)(branch_4)
  branch_4 = BatchNormalization()(branch_4)

  CEM_out = Add()([branch_1,branch_2,branch_3,branch_4])

  return CEM_out


def l1(Input,K):
  
  C_0 = CEM(Input,K)

  C_1 = CEM(Input,K)

  C_2 = CEM(Input,K)

  return C_0,C_1,C_2


def IM(CM,K):
  
  branch_1 = Conv2D(64, (K[0], 1), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(CM)
  branch_1 = Dropout(0.1)(branch_1)
  branch_1 = BatchNormalization()(branch_1)

  branch_2 = Conv2D(64, (K[1], 1), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(CM)
  branch_2 = Conv2D(64, (1,K[1]), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(branch_2)
  branch_2 = Dropout(0.1)(branch_2)
  branch_2 = BatchNormalization()(branch_2)

  branch_3 = Conv2D(64, (K[2], 1), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(CM)
  branch_3 = Conv2D(64, (1,K[2]), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(branch_3)
  branch_3 = Dropout(0.1)(branch_3)
  branch_3 = BatchNormalization()(branch_3)

  IM_out = Add()([branch_1,branch_2,branch_3])

  return IM_out


def l2(C_0,C_1,C_2,K):

  e_1 = MaxPool2D((8,8), strides = 8, padding='same')(C_0)

  C_1 = MaxPool2D((2,2), strides = 2, padding='same')(C_1)
  I_0 = IM(C_1,K)

  C_2 = MaxPool2D((2,2), strides = 2, padding='same')(C_2)
  I_1 = IM(C_2,K)

  return e_1,I_0,I_1


def LEM(Input,K):

  branch_1 = Conv2D(128, (K[0], 1), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(Input)
  branch_1 = Dropout(0.1)(branch_1)
  branch_1 = BatchNormalization()(branch_1)

  branch_2 = Conv2D(128, (K[1], 1), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(Input)
  branch_2 = Conv2D(128, (1,K[1]), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(Input)
  branch_2 = Dropout(0.1)(branch_2)
  branch_2 = BatchNormalization()(branch_2)

  LEM_out = Add()([branch_1,branch_2])

  return LEM_out


def l3(I_0,I_1,K):

  e_2 = MaxPool2D((4,4), strides = 4, padding='same')(I_0)


  I_1 = MaxPool2D((2,2), strides = 2, padding='same')(I_1)
  L_0 = LEM(I_1,K)
  e_3 = MaxPool2D((2,2), strides = 2, padding='same')(L_0)

  return e_2,L_0,e_3


def MultiscaleFeatureFusion(e_1,e_2,e_3):

  MFF = tf.concat([e_1,e_2,e_3], axis=3)

  return MFF


def GuideBlock(Input,filters,k):
  branch_1 = Conv2D(filters, (1,k[1]), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(Input)
  branch_1 = Conv2D(filters, (k[1],1), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(branch_1)
  branch_1 = Dropout(0.1)(branch_1)
  branch_1 = BatchNormalization()(branch_1)

  branch_2 = Conv2D(filters, (1,k[1]), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(Input)
  branch_2 = Conv2D(filters, (k[1],1), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(branch_2)
  branch_2 = Dropout(0.1)(branch_2)
  branch_2 = BatchNormalization()(branch_2)

  GB_out = Add()([branch_1,branch_2])
  GB_out = Conv2D(filters, (1,k[0]), activation=actv_funtion, kernel_initializer='he_normal', padding='same',strides = 1)(GB_out)
  GB_out = Dropout(0.1)(GB_out)
  GB_out = BatchNormalization()(GB_out)

  return GB_out


#guide https://www.sciencedirect.com/science/article/pii/S0893608019302503?via%3Dihub
#inception https://media5.datahacker.rs/2018/11/inception_module.png
def DecoderBlock(C_0,I_0,L_0,MFF,K,type_op):

  if type_op == 'convtranspose':

    MFF_up = Conv2DTranspose(128, (2, 2),activation=actv_funtion, kernel_initializer='he_normal', strides=2)(MFF)
    MFF_up = Dropout(0.1)(MFF_up)
    MFF_up = BatchNormalization()(MFF_up)
    
    GB_1 = GuideBlock(MFF_up,128,K)

    OBG_L = Multiply()([GB_1,L_0])

    X_l = concatenate([OBG_L,MFF_up])

    X_l_up = Conv2DTranspose(64, (2, 2),activation=actv_funtion, kernel_initializer='he_normal', strides=2)(X_l)
    X_l_up = Dropout(0.1)(X_l_up)
    X_l_up = BatchNormalization()(X_l_up)
    GB_2 = GuideBlock(X_l_up,64,K)

    OBG_I = Multiply()([GB_2,I_0])

    X_2 = concatenate([OBG_I,X_l_up])

    X_2_up =  Conv2DTranspose(32, (2, 2),activation=actv_funtion, kernel_initializer='he_normal', strides=2)(X_2)
    X_2_up = Dropout(0.1)(X_2_up)
    X_2_up = BatchNormalization()(X_2_up)
    GB_3 = GuideBlock(X_2_up,32,K)

    OBG_C = Multiply()([GB_3,C_0])

  if type_op == 'uptranspose':

    MFF_up = UpSampling2D(size=(2,2))(MFF)
    GB_1 = GuideBlock(MFF_up,128,K)

    OBG_L = Multiply()([GB_1,L_0])


    X_l = concatenate([OBG_L,MFF_up])

    X_l_up = UpSampling2D(size=(2,2))(X_l)
    GB_2 = GuideBlock(X_l_up,64,K)

    OBG_I = Multiply()([GB_2,I_0])


    X_2 = concatenate([OBG_I,X_l_up])

    X_2_up =  UpSampling2D(size=(2,2))(X_2)
    GB_3 = GuideBlock(X_2_up,32,K)

    OBG_C = Multiply()([GB_3,C_0])


  out_put = concatenate([OBG_C,X_2_up])
  out_put = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(out_put)

  return out_put


def MF2Net(K,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,type_op_decoder_block):

  input = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

  C_0,C_1,C_2 = l1(input,K)
  e_1,I_0,I_1 = l2(C_0,C_1,C_2,K)
  e_2,L_0,e_3 = l3(I_0,I_1,K)

  MFF = MultiscaleFeatureFusion(e_1,e_2,e_3)

  output = DecoderBlock(C_0,I_0,L_0,MFF,K,type_op_decoder_block)

  MF2_model = Model(inputs=input, outputs=output,name='MF2_Net')

  return MF2_model

def CreateModel(K,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,type_op_decoder_block, summa: bool):
  
  model_MF2Net = MF2Net(K,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,type_op_decoder_block)
  
  if summa:
    model_MF2Net.summary()  

  return model_MF2Net
  