import numpy as np
from PIL import Image
from utils import FDA_source_to_target
import torch
import older_scipy
import os
import random
source_directory = 'prova\\'
target_directory = 'night'
save_directory = 'night_train'
#target_images = os.listdir()



targets_images = os.listdir(target_directory)

def transform(source: str, target, dir):
    im_src = Image.open("demo_images/day.jpg").convert('RGB')
    im_trg = Image.open("demo_images/night.jpg").convert('RGB')

    w,h = im_src.size


    im_src = im_src.resize( (1024,512), Image.BICUBIC )
    im_trg = im_trg.resize( (1024,512), Image.BICUBIC )

    im_src = np.asarray(im_src, np.float32)
    im_trg = np.asarray(im_trg, np.float32)

    im_src = im_src.transpose((2, 0, 1))
    im_trg = im_trg.transpose((2, 0, 1))

    im_src = torch.from_numpy(im_src).unsqueeze(0)
    im_trg = torch.from_numpy(im_trg).unsqueeze(0)

    src_in_trg = FDA_source_to_target( im_src, im_trg, L=0.01 )

    src_in_trg = torch.Tensor.numpy(src_in_trg.squeeze(0))

    src_in_trg = src_in_trg.transpose((1,2,0))
    src_in_trg = src_in_trg.astype(int)
    #Image.fromarray(np.uint8(src_in_trg)).save('demo_images/src_in_tar.png')
    name = source.split('/')[-1].replace('@','night@',1)
    older_scipy.toimage(src_in_trg, cmin=0.0, cmax=255.0).resize((w,h)).save(f'{dir}/{name}')




for dir in os.listdir(source_directory):
    dir_path = os.path.join(source_directory,dir)
    os.mkdir(os.path.join(save_directory,dir))
    for file in os.listdir(dir_path):
        random_index = int(random.uniform(0,105))
        transform(os.join.path(dir_path,file),os.join.path(target_directory,targets_images[random_index]),os.path.join(save_directory,dir))

        



