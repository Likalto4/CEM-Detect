# add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None


import cv2 as cv
from tqdm import tqdm
import numpy as np
import pandas as pd

def main():

    # information
    bbox_small_df = pd.read_csv(repo_path / 'data/CDD-CESM/masks/bbox_small_masses.csv')
    im_dir = repo_path / 'data/CDD-CESM/images/substracted'

    # saving paths
    instance = 'CEM-square_small_bbox'
    saving_im_dir = repo_path / 'data/CDD-CESM/masses_closeup/images' / instance
    saving_mask_dir = repo_path / 'data/CDD-CESM/masses_closeup/masks' / instance
    saving_im_dir.mkdir(parents=True, exist_ok=True)
    saving_mask_dir.mkdir(parents=True, exist_ok=True)


    # loop on rows
    for row_idx in tqdm(range(len(bbox_small_df))):
        ex_row = bbox_small_df.iloc[row_idx]
        # image
        im_path = im_dir /  (ex_row['image_name'] + '.jpg')
        im = cv.imread(str(im_path))
        im_x_max, im_y_max = im.shape[1], im.shape[0]
        # read mask image
        mask_path = repo_path / 'data/CDD-CESM/masks/substracted' / f"{ex_row['image_name']}_reg{ex_row['region_id']}.png"
        mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

        #bbox math
        current_x, current_y, current_w, current_h = eval(ex_row['bbox'])
        
        # expanding requirement
        expand_w = 512 - current_w
        expand_h = 512 - current_h
        print(expand_w, expand_h)
        
        # expand the bbox on both sides, unevenly following a random ratio
        left_expand = np.random.randint(0, expand_w) if expand_w > 0 else 0
        right_expand = expand_w - left_expand
        top_expand = np.random.randint(0, expand_h) if expand_h > 0 else 0
        bottom_expand = expand_h - top_expand

        # create new bbox
        new_x = current_x - left_expand
        new_y = current_y - top_expand
        new_w = current_w + left_expand + right_expand
        new_h = current_h + top_expand + bottom_expand
        # check if the new bbox is out of the image
        if new_x < 0:
            new_x = 0
        if new_y < 0:
            new_y = 0
        if new_x + new_w > im_x_max:
            new_x = im_x_max - new_w
        if new_y + new_h > im_y_max:
            new_y = im_y_max - new_h


        # save new image and mask coordinates
        new_im = im[new_y:new_y+new_h, new_x:new_x+new_w]
        new_mask = mask[new_y:new_y+new_h, new_x:new_x+new_w]
        # convert the original shape free binary mask to a bbox square mask
        x, y, w, h = cv.boundingRect(new_mask)
        new_mask = np.zeros_like(new_mask)
        new_mask[y:y+h, x:x+w] = 255

        # save the new image and mask
        saving_im_path = saving_im_dir / f"{ex_row['image_name']}_reg{ex_row['region_id']}.jpg"
        saving_mask_path = saving_mask_dir / f"{ex_row['image_name']}_reg{ex_row['region_id']}.png"
        cv.imwrite(str(saving_im_path), new_im)
        cv.imwrite(str(saving_mask_path), new_mask)


if __name__ == "__main__":
    main()