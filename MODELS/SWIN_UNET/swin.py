#adativef https://github.com/yingkaisha/keras-unet-collection/blob/main/keras_unet_collection/_model_swin_unet_2d.py
#https://arxiv.org/abs/2105.05537


from __future__ import absolute_import

from layer_utils import *
from transformer_layers import patch_extract, patch_embedding, SwinTransformerBlock, patch_merging, patch_expanding

from tensorflow.keras.layers import Input, Dense,concatenate,Activation
from tensorflow.keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, MaxPool2D, Conv2DTranspose, concatenate


def CONV_output(X, n_labels, kernel_size=1, activation='Softmax', name='conv_output'):
  X = Conv2D(n_labels, kernel_size, padding='same', use_bias=True, name=name)(X)
  
  if activation:
    if activation == 'Sigmoid':
      X = Activation('sigmoid', name='{}_activation'.format(name))(X)
    else:
      activation_func = eval(activation)
      X = activation_func(name='{}_activation'.format(name))(X)
            
  return X

def swin_transformer_stack(X, stack_num, embed_dim, num_patch, num_heads, window_size, num_mlp, shift_window=True, name=''):

  # Turn-off dropouts
  mlp_drop_rate = 0 # Droupout after each MLP layer
  attn_drop_rate = 0 # Dropout after Swin-Attention
  proj_drop_rate = 0 # Dropout at the end of each Swin-Attention block, i.e., after linear projections
  drop_path_rate = 0 # Drop-path within skip-connections
  
  qkv_bias = True # Convert embedded patches to query, key, and values with a learnable additive value
  qk_scale = None # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor
  
  if shift_window:
      shift_size = window_size // 2
  else:
      shift_size = 0
  
  for i in range(stack_num):
  
      if i % 2 == 0:
          shift_size_temp = 0
      else:
          shift_size_temp = shift_size

      X = SwinTransformerBlock(dim=embed_dim, num_patch=num_patch, num_heads=num_heads, 
                                window_size=window_size, shift_size=shift_size_temp, num_mlp=num_mlp, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                mlp_drop=mlp_drop_rate, attn_drop=attn_drop_rate, proj_drop=proj_drop_rate, drop_path_prob=drop_path_rate, 
                                name='name{}'.format(i))(X)
  return X


def swin_unet_2d_base(input_tensor, filter_num_begin, depth, stack_num_down, stack_num_up, 
                      patch_size, num_heads, window_size, num_mlp, shift_window=True, name='swin_unet'):

  # Compute number be patches to be embeded
  input_size = input_tensor.shape.as_list()[1:]
  num_patch_x = input_size[0]//patch_size[0]
  num_patch_y = input_size[1]//patch_size[1]
  
  # Number of Embedded dimensions
  embed_dim = filter_num_begin
  
  depth_ = depth
  
  X_skip = []

  X = input_tensor
  
  # Patch extraction
  X = patch_extract(patch_size)(X)

  # Embed patches to tokens
  X = patch_embedding(num_patch_x*num_patch_y, embed_dim)(X)
  
  # The first Swin Transformer stack
  X = swin_transformer_stack(X, stack_num=stack_num_down, 
                              embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y), 
                              num_heads=num_heads[0], window_size=window_size[0], num_mlp=num_mlp, 
                              shift_window=shift_window, name='{}_swin_down0'.format(name))
  X_skip.append(X)
  
  # Downsampling blocks
  for i in range(depth_-1):
      
    # Patch merging
    X = patch_merging((num_patch_x, num_patch_y), embed_dim=embed_dim, name='down{}'.format(i))(X)
    
    # update token shape info
    embed_dim = embed_dim*2
    num_patch_x = num_patch_x//2
    num_patch_y = num_patch_y//2
    
    # Swin Transformer stacks
    X = swin_transformer_stack(X, stack_num=stack_num_down, 
                                embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y), 
                                num_heads=num_heads[i+1], window_size=window_size[i+1], num_mlp=num_mlp, 
                                shift_window=shift_window, name='{}_swin_down{}'.format(name, i+1))
    
    # Store tensors for concat
    X_skip.append(X)
      
  # reverse indexing encoded tensors and hyperparams
  X_skip = X_skip[::-1]
  num_heads = num_heads[::-1]
  window_size = window_size[::-1]
  
  # upsampling begins at the deepest available tensor
  X = X_skip[0]
  
  # other tensors are preserved for concatenation
  X_decode = X_skip[1:]
  
  depth_decode = len(X_decode)
  
  for i in range(depth_decode):
    # Patch expanding
    X = patch_expanding(num_patch=(num_patch_x, num_patch_y),
                          embed_dim=embed_dim, upsample_rate=2, return_vector=True, name='{}_swin_up{}'.format(name, i))(X)
      

      # update token shape info
    embed_dim = embed_dim//2
    num_patch_x = num_patch_x*2
    num_patch_y = num_patch_y*2
      
      # Concatenation and linear projection
    X = concatenate([X, X_decode[i]], axis=-1, name='{}_concat_{}'.format(name, i))
    X = Dense(embed_dim, use_bias=False, name='{}_concat_linear_proj_{}'.format(name, i))(X)
      
      # Swin Transformer stacks
    X = swin_transformer_stack(X, stack_num=stack_num_up, 
                          embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y), 
                          num_heads=num_heads[i], window_size=window_size[i], num_mlp=num_mlp, 
                          shift_window=shift_window, name='{}_swin_up{}'.format(name, i))
      
  # The last expanding layer; it produces full-size feature maps based on the patch size
  # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
  X = patch_expanding(num_patch=(num_patch_x, num_patch_y),
                      embed_dim=embed_dim, upsample_rate=patch_size[0], return_vector=False)(X)
  
  return X


def swin_unet_2d(input_size, filter_num_begin, n_labels, depth, stack_num_down, stack_num_up, 
                      patch_size, num_heads, window_size, num_mlp, output_activation='Softmax', shift_window=True, name='Swin_unet'):

  IN = Input(input_size)
  
  # base    
  X = swin_unet_2d_base(IN, filter_num_begin=filter_num_begin, depth=depth, stack_num_down=stack_num_down, stack_num_up=stack_num_up, 
                        patch_size=patch_size, num_heads=num_heads, window_size=window_size, num_mlp=num_mlp, shift_window=shift_window, name=name)
  
  # output layer
  OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))
  
  # functional API model
  model = Model(inputs=[IN,], outputs=[OUT,], name='{}_model'.format(name))
    
  return model

def CreateModel(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, filter_num_begin, n_labels, depth, stack_num_down, stack_num_up,patch_size, num_heads, window_size, num_mlp, summa: bool):
  inp = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
  
  model_swin_unet_2d = swin_unet_2d(inp , filter_num_begin, n_labels, depth, stack_num_down, stack_num_up,patch_size, num_heads, window_size, num_mlp)
  
  if summa:
    model_swin_unet_2d.summary()  
    
  return model_swin_unet_2d
