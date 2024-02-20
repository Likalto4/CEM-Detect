from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None
exp_path = Path.cwd().resolve() # experiment path
# visible GPUs
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image

import SimpleITK as sitk
from tqdm import tqdm
from IPython.display import clear_output

def registrater_affine(fixed:sitk.Image, moving:sitk.Image, show=False):
    """registration of two sitk images using affine transformation

    Args:
        fixed (sitk.Image): fixed image. In CEDM SET, it is the image with contrast enhancement
        moving (sitk.Image): moving image, in CEDM is the image before contrast injection

    Returns:
        sitk.ParameterMap: transform parameters
    """
    # register
    parameterMap = sitk.GetDefaultParameterMap('affine')
    parameterMap['NumberOfSpatialSamples'] = ['5000']
    #run registration
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(moving)
    elastixImageFilter.SetParameterMap(parameterMap)
    elastixImageFilter.Execute()
    moved_im = elastixImageFilter.GetResultImage()
    #save transform parameters
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()

    clear_output(wait=show)

    return transformParameterMap, moved_im

def main():
    train_metadata_path = repo_path / 'data/CDD-CESM/metadata/bboxes/split_1' / 'test_set.csv' # <--- change this
    im_dir_CEM = repo_path / 'data/CDD-CESM/images/substracted'
    im_dir_low = repo_path / 'data/CDD-CESM/images/low-energy'
    train_metadata = pd.read_csv(train_metadata_path)

    # saving dir
    save_dir = repo_path / 'generation/bbridge/data/CEM-512/split_1' / 'val' # <--- change this
    save_dir.mkdir(parents=True, exist_ok=True)

    # loop on images
    for image_name in tqdm(train_metadata['image_name'].unique()):
        # image_name = train_metadata['image_name'][20]
        ex_im_path = im_dir_CEM / (image_name + '.jpg')
        # print(f'Processing image: {ex_im_path.stem}')
        low_im_path = im_dir_low / (ex_im_path.stem.replace('CM', 'DM') + '.jpg')

        # read the image
        ex_im = cv.imread(str(ex_im_path), cv.IMREAD_GRAYSCALE)
        low_im = cv.imread(str(low_im_path), cv.IMREAD_GRAYSCALE)
        print(f'image shapes: {ex_im.shape}, {low_im.shape}')

        # register images
        fixedImage = sitk.GetImageFromArray(ex_im)
        movingImage = sitk.GetImageFromArray(low_im)
        _, moved_im = registrater_affine(fixedImage, movingImage, show=False)
        low_im = sitk.GetArrayFromImage(moved_im)

        print(f'Shape after registration: {ex_im.shape}, {low_im.shape}')

        # create a subimage of 512x512, starting from the 0,0 corner
        for i_y in range(len(ex_im)//512):
            for i_x in range(len(ex_im[0])//512):
                y_shift = 512*i_y
                x_shift = 512*i_x

                from_y = 0+y_shift
                to_y = 512+y_shift
                from_x = 0+x_shift
                to_x = 512+x_shift

                if to_y > ex_im.shape[0]: # when to_y is too large (end of image)
                    # to_complete_y = 512 - (ex_im.shape[0] - from_y)
                    to_y = ex_im.shape[0]

                    sub_im = np.zeros((512, 512), dtype=np.uint8)
                    # fill the subimage with the original image
                    sub_im[:to_y-from_y, :] = ex_im[from_y:to_y, from_x:to_x]
                    # replicate for low energy image
                    sub_low_im = np.zeros((512, 512), dtype=np.uint8)
                    sub_low_im[:to_y-from_y, :] = low_im[from_y:to_y, from_x:to_x]
                elif to_x > ex_im.shape[1]:
                    # to_complete_x = 512 - (ex_im.shape[1] - from_x)
                    to_x = ex_im.shape[1]

                    sub_im = np.zeros((512, 512), dtype=np.uint8)
                    # fill the subimage with the original image
                    sub_im[:, :to_x-from_x] = ex_im[from_y:to_y, from_x:to_x]
                    # replicate for low energy image
                    sub_low_im = np.zeros((512, 512), dtype=np.uint8)
                    sub_low_im[:, :to_x-from_x] = low_im[from_y:to_y, from_x:to_x]

                else: # when limits are within the image
                    sub_im = ex_im[from_y:to_y, from_x:to_x]
                    sub_low_im = low_im[from_y:to_y, from_x:to_x]

                # if more than 30% of the image is black, then the image is considered black
                black = np.sum(sub_im == 0) / (512*512) > 0.7
                sub_im = Image.fromarray(sub_im)
                sub_low_im = Image.fromarray(sub_low_im).convert('L')
                print(f'Image from y:{from_y} to y:{to_y} and x:{from_x} to x:{to_x} is black: {black}')

                # #### register images
                # fixedImage = sitk.GetImageFromArray(np.array(sub_im))
                # movingImage = sitk.GetImageFromArray(np.array(sub_low_im))
                # _, moved_im = registrater_affine(fixedImage, movingImage, show=False)
                # sub_low_im = Image.fromarray(sitk.GetArrayFromImage(moved_im)).convert('L')
                # #####

                # save image in the save_dir
                save_path = save_dir / 'B' / f'{ex_im_path.stem}_y{i_y}_x{i_x}.jpg'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                sub_im.save(save_path)

                save_path_low = save_dir / 'A' / f'{low_im_path.stem}_y{i_y}_x{i_x}.jpg'
                save_path_low.parent.mkdir(parents=True, exist_ok=True)
                sub_low_im.save(save_path_low)

if __name__ == "__main__":
    main()