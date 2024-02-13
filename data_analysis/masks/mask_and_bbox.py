#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

from utils import dataset_CDD_CESM, patient_CDD

import numpy as np
import cv2 as cv
import pandas as pd
from tqdm import tqdm

def generate_ellipse_mask(image_shape:tuple, center_xy:tuple, axes:tuple):
    """generates a binary mask of an ellipse, given the original image dimensions and the ellipse parameters

    Args:
        image_shape (tuple): (height, width) of the original image
        center_xy (tuple): (center_x, center_y) of the ellipse
        axes (tuple): (a, b) main and minor axes of the ellipse

    Returns:
        np.array: binary 0,255 numpy array
    """
    height, width = image_shape[:2]
    center_x, center_y = center_xy
    a, b = axes
    y, x = np.ogrid[:height, :width]
    ellipse_mask = (((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2) <= 1
    
    return ellipse_mask.astype(np.uint8) * 255

def generate_polygon_mask(image_shape:tuple, vertices:np.array):
    """generates a binary mask of a polygon, given the original image dimensions and the polygon vertices

    Args:
        image_shape (tuple): (height, width) of the original image
        vertices (np.array): array of shape (n,2) of the polygon vertices

    Returns:
        np.array: binary 0,255 numpy array
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv.fillPoly(mask, [vertices], 255)
    return mask

# create bbox coordinates around the binary mask
def bbox_from_mask(mask:np.array):
    """returns the coordinates of the bounding box around the binary mask

    Args:
        mask (np.array): binary mask

    Returns:
        tuple: (x,y,w,h) coordinates of the bounding box
    """
    x, y, w, h = cv.boundingRect(mask)
    return (x,y,w,h)

def main():
    mode = 'substracted'
    dataset_CESM = dataset_CDD_CESM(mode=mode)
    print(dataset_CESM)

    excluding_list = ['non enhancement', 'focus enhancement', 'rim enhancement']

    # exclude cases with 'non mass' tags
    meta_filterd = dataset_CESM.metadata[~dataset_CESM.metadata['Tags'].str.contains('non mass', case=False)]
    for case in excluding_list:
        meta_filterd = meta_filterd[~meta_filterd['Findings'].str.contains(case, case=False)]

    # filter to keep only casses with word 'mass' in the tags
    meta_filterd = meta_filterd[meta_filterd['Tags'].str.contains('mass')]
    print(meta_filterd['Pathology Classification/ Follow up'].value_counts().to_string())
    mass_patients = meta_filterd['Patient_ID'].unique()
    print(f'Number of patients with mass: {len(mass_patients)}')
    dataset_CESM.redefine_metadata(meta_filterd)

    masks_dir = repo_path / 'data/CDD-CESM/masks'
    bbox_dataframe = None

    for pat_id in tqdm(dataset_CESM.patient_ids):
        patient = patient_CDD(pat_id, dataset_CESM)
        # print(patient)
        while True:
            patient.set_image(show_status=False) # to load the image info
            image = patient.get_array(flip=False, plot=False)
            # print(f'The current image has {patient.image_num_annotations} annotations')
            # loop of lesions
            if patient.image_num_annotations!=0: # check if there are annotations
                for region_num in range(patient.image_num_annotations):
                    mask = None # reset mask, just to be sure not to use the previous one !
                    dic_ex = patient.image_annotations[patient.image_annotations.region_id==region_num].region_shape_attributes.values[0]
                    # print(f'The annotation is a(n) {dic_ex["name"]}')
                    if dic_ex['name'] in ['ellipse','circle']:
                        center, axes = patient.ellipse_reader(dic_ex)
                        mask = generate_ellipse_mask(image.shape, center, axes)
                    elif dic_ex['name']=='polygon' or dic_ex['name']=='polyline':
                        vertices = patient.polygon_reader(dic_ex)
                        mask = generate_polygon_mask(image.shape, vertices)
                    elif dic_ex['name']=='point':
                        print('Point annotations are not supported.')
                        print(patient)
                        continue

                    # save mask
                    saving_dir = masks_dir / patient.image_path.parent.name /  f'{patient.image_path.stem}_reg{region_num}.png'
                    saving_dir.parent.mkdir(parents=True, exist_ok=True)
                    cv.imwrite(str(saving_dir), mask)
                    # save bbox
                    bbox_actual = pd.DataFrame({'patient_id':[pat_id], 'image_name': [patient.image_path.stem], 'region_id': [region_num], 'bbox': [bbox_from_mask(mask)]})
                    bbox_dataframe = pd.concat([bbox_dataframe, bbox_actual], ignore_index=True)

            if patient.row_counter==-1:
                        break
    # save bbox dataframe
    bbox_dataframe.to_csv(masks_dir / 'bbox_CESM.csv', index=False)

if __name__ == "__main__":
    main()
