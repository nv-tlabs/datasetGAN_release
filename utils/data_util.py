"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np

face_class = ['background', 'head', 'head***cheek', 'head***chin', 'head***ear', 'head***ear***helix',
              'head***ear***lobule', 'head***eye***botton lid', 'head***eye***eyelashes', 'head***eye***iris',
              'head***eye***pupil', 'head***eye***sclera', 'head***eye***tear duct', 'head***eye***top lid',
              'head***eyebrow', 'head***forehead', 'head***frown', 'head***hair', 'head***hair***sideburns',
              'head***jaw', 'head***moustache', 'head***mouth***inferior lip', 'head***mouth***oral comisure',
              'head***mouth***superior lip', 'head***mouth***teeth', 'head***neck', 'head***nose',
              'head***nose***ala of nose', 'head***nose***bridge', 'head***nose***nose tip', 'head***nose***nostril',
              'head***philtrum', 'head***temple', 'head***wrinkles']

car_12_class = ['background', 'car_body', 'head light', 'tail light', 'licence plate',
                'wind shield', 'wheel', 'door', 'handle' , 'wheelhub', 'window', 'mirror']
car_20_class = ['background', 'back_bumper', 'bumper', 'car_body', 'car_lights', 'door', 'fender','grilles','handles',
                'hoods', 'licensePlate', 'mirror','roof', 'running_boards', 'tailLight','tire', 'trunk_lids','wheelhub', 'window', 'windshield']


car_20_palette =[ 255,  255,  255, # 0 background
  238,  229,  102,# 1 back_bumper
  0, 0, 0,# 2 bumper
  124,  99 , 34, # 3 car
  193 , 127,  15,# 4 car_lights
  248  ,213 , 42, # 5 door
  220  ,147 , 77, # 6 fender
  99 , 83  , 3, # 7 grilles
  116 , 116 , 138,  # 8 handles
  200  ,226 , 37, # 9 hoods
  225 , 184 , 161, # 10 licensePlate
  142 , 172  ,248, # 11 mirror
  153 , 112 , 146, # 12 roof
  38  ,112 , 254, # 13 running_boards
  229 , 30  ,141, # 14 tailLight
  52 , 83  ,84, # 15 tire
  194 , 87 , 125, # 16 trunk_lids
  225,  96  ,18,  # 17 wheelhub
  31 , 102 , 211, # 18 window
  104 , 131 , 101# 19 windshield
         ]



face_palette = [  1.0000,  1.0000 , 1.0000,
              0.4420,  0.5100 , 0.4234,
              0.8562,  0.9537 , 0.3188,
              0.2405,  0.4699 , 0.9918,
              0.8434,  0.9329  ,0.7544,
              0.3748,  0.7917 , 0.3256,
              0.0190,  0.4943 , 0.3782,
              0.7461 , 0.0137 , 0.5684,
              0.1644,  0.2402 , 0.7324,
              0.0200 , 0.4379 , 0.4100,
              0.5853 , 0.8880 , 0.6137,
              0.7991 , 0.9132 , 0.9720,
              0.6816 , 0.6237  ,0.8562,
              0.9981 , 0.4692 , 0.3849,
              0.5351 , 0.8242 , 0.2731,
              0.1747 , 0.3626 , 0.8345,
              0.5323 , 0.6668 , 0.4922,
              0.2122 , 0.3483 , 0.4707,
              0.6844,  0.1238 , 0.1452,
              0.3882 , 0.4664 , 0.1003,
              0.2296,  0.0401 , 0.3030,
              0.5751 , 0.5467 , 0.9835,
              0.1308 , 0.9628,  0.0777,
              0.2849  ,0.1846 , 0.2625,
              0.9764 , 0.9420 , 0.6628,
              0.3893 , 0.4456 , 0.6433,
              0.8705 , 0.3957 , 0.0963,
              0.6117 , 0.9702 , 0.0247,
              0.3668 , 0.6694 , 0.3117,
              0.6451 , 0.7302,  0.9542,
              0.6171 , 0.1097,  0.9053,
              0.3377 , 0.4950,  0.7284,
              0.1655,  0.9254,  0.6557,
              0.9450  ,0.6721,  0.6162]

face_palette = [int(item * 255) for item in face_palette]





car_12_palette =[ 255,  255,  255, # 0 background
         124,  99 , 34, # 3 car
         193 , 127,  15,# 4 car_lights
         229 , 30  ,141, # 14 tailLight
        225 , 184 , 161, # 10 licensePlate
        104 , 131 , 101,# 19 windshield
        52 , 83  ,84, # 15 tire
        248  ,213 , 42, # 5 door
         116 , 116 , 138,  # 8 handles
           225,  96  ,18,  # 17 wheelhub
         31 , 102 , 211, # 18 window
         142 , 172  ,248, # 11 mirror
         ]



car_32_palette =[ 255,  255,  255,
  238,  229,  102,
  0, 0, 0,
  124,  99 , 34,
  193 , 127,  15,
  106,  177,  21,
  248  ,213 , 42,
  252 , 155,  83,
  220  ,147 , 77,
  99 , 83  , 3,
  116 , 116 , 138,
  63  ,182 , 24,
  200  ,226 , 37,
  225 , 184 , 161,
  233 ,  5  ,219,
  142 , 172  ,248,
  153 , 112 , 146,
  38  ,112 , 254,
  229 , 30  ,141,
  115  ,208 , 131,
  52 , 83  ,84,
  229 , 63 , 110,
  194 , 87 , 125,
  225,  96  ,18,
  73  ,139,  226,
  172 , 143 , 16,
  169 , 101 , 111,
  31 , 102 , 211,
  104 , 131 , 101,
  70  ,168  ,156,
  183 , 242 , 209,
  72  ,184 , 226]

bedroom_palette =[ 255,  255,  255,
  238,  229,  102,
  255, 72, 69,
  124,  99 , 34,
  193 , 127,  15,
  106,  177,  21,
  248  ,213 , 42,
  252 , 155,  83,
  220  ,147 , 77,
  99 , 83  , 3,
  116 , 116 , 138,
  63  ,182 , 24,
  200  ,226 , 37,
  225 , 184 , 161,
  233 ,  5  ,219,
  142 , 172  ,248,
  153 , 112 , 146,
  38  ,112 , 254,
  229 , 30  ,141,
   238, 229, 12,
   255, 72, 6,
   124, 9, 34,
   193, 17, 15,
   106, 17, 21,
   28, 213, 2,
   252, 155, 3,
   20, 147, 77,
   9, 83, 3,
   11, 16, 138,
   6, 12, 24,
   20, 22, 37,
   225, 14, 16,
   23, 5, 29,
   14, 12, 28,
   15, 11, 16,
   3, 12, 24,
   22, 3, 11
   ]

cat_palette = [255,  255,  255,
            220, 220, 0,
           190, 153, 153,
            250, 170, 30,
           220, 220, 0,
           107, 142, 35,
           102, 102, 156,
           152, 251, 152,
           119, 11, 32,
           244, 35, 232,
           220, 20, 60,
           52 , 83  ,84,
          194 , 87 , 125,
          225,  96  ,18,
          31 , 102 , 211,
          104 , 131 , 101
          ]

def trans_mask_stylegan_20classTo12(mask):
    final_mask = np.zeros(mask.shape)
    final_mask[(mask != 0)] = 1 # car
    final_mask[(mask == 4)] = 2 # head light
    final_mask[(mask == 14)] = 5 # tail light
    final_mask[(mask == 10)] = 3 # licence plate
    final_mask[ (mask == 19)] = 8 # wind shield
    final_mask[(mask == 15)] = 6 # wheel
    final_mask[(mask == 5)] = 9 # door
    final_mask[(mask == 8)] = 10 # handle
    final_mask[(mask == 17)] = 11 # wheelhub
    final_mask[(mask == 18)] = 7 # window
    final_mask[(mask == 11)] = 4 # mirror
    return final_mask


def trans_mask(mask):
    return mask
