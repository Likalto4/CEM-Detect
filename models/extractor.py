# add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import torch
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.7")

import numpy as np
import cv2 as cv
import pandas as pd
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import Boxes

setup_logger()

class DBT_extractor():
    """Class to extract the backbone features from an image
    """
    def __init__(self, config_file:str, model_file:str, min_score:float):
        """Initialize the class

        Args:
            config_file (str): path to the config file
            model_file (str): path to the model file
            min_score (float): minimum score to keep a prediction
        """
        self.config_file = config_file
        self.model_file = model_file
        self.min_score = min_score
        self.predictor = self._initialize_predictor()
        self.main_df = None
        self.main_df_path = None
        self.feature_name = None
        
        
    def _initialize_predictor(self):
        """Initialize the predictor

        Returns:
            detectron2.engine.DefaultPredictor: predictor
        """
        cfg = get_cfg()
        cfg.merge_from_file(self.config_file)
        cfg.MODEL.WEIGHTS = self.model_file
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.min_score  # set the testing threshold for this model
        predictor = DefaultPredictor(cfg)
        return predictor
    
    # when printing
    def __repr__(self):
        return f'DBT_extractor(config_file={self.config_file}, model_file={self.model_file}, min_score={self.min_score})'

    def get_normal_BBox (self, im_array:np.array):
        """Given an mammogram image, returns the bounding box of the breast

        Args:
            im_array (np.array): array of the mammogram image, with background black

        Returns:
            tuple, array: bounding box coordinates, and image with the breast only
        """
        #threshold im_array 
        img = cv.threshold(im_array, 0, 255, cv.THRESH_BINARY)[1]  # ensure binary
        nb_components, output, stats, _ = cv.connectedComponentsWithStats(img, connectivity=4)
        sizes = stats[:, -1]
        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]
        img2 = np.zeros(output.shape,dtype=np.uint8)
        img2[output == max_label] = 255
        contours, _ = cv.findContours(img2,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

        cnt = contours[0]

        x,y,w,h = cv.boundingRect(cnt)
        
        return (x,y,x+w,y+h), img2

    def prepare_bbox(self, bbox_lesion:np.array, predictor:detectron2.engine.defaults.DefaultPredictor, image_rgb:np.array):
        """Transform bbox to the format of the backbone

        Args:
            bbox_lesion (np.array): bbox in the format [x1, y1, x2, y2]
            predictor (detectron2.engine.defaults.DefaultPredictor): predictor, to know the augmentation technique
            image_rgb (np.array): image, to know the resizing

        Returns:
            Boxes: transformed bbox
        """
        # transform bbox to the format of the backbone
        new_bbox_lesion = predictor.aug.get_transform(image_rgb).apply_box([bbox_lesion])
        new_bbox_lesion = torch.as_tensor(new_bbox_lesion).cuda()
        # transform to boxes object
        new_bbox_lesion = Boxes(new_bbox_lesion)
        assert new_bbox_lesion.tensor.shape == (1, 4)
        
        return new_bbox_lesion

    def backbone_feature_extraction(self, predictor:detectron2.engine.DefaultPredictor, image_rgb:np.array):
        """Extract the backbone features from the image

        Args:
            predictor (detectron2.engine.DefaultPredictor): default predictor
            image_rgb (np.array): image to extract the features from

        Returns:
            dict: dictionary with the feature maps, p2 to p6
        """
        ### PYRAMID FEATURES
        with torch.no_grad():
            height, width = image_rgb.shape[:2]
            image = predictor.aug.get_transform(image_rgb).apply_image(image_rgb)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            images = predictor.model.preprocess_image([inputs]) # additional preprocessing step

            feature_maps = predictor.model.backbone(images.tensor)

        return feature_maps

    def bbox_pooler_feature_extraction(self, predictor:detectron2.engine.DefaultPredictor, feature_maps:dict, new_bbox_lesion:Boxes):
        """Extract the features from the bbox using the head pooler, activily using the bbox to crop the feature maps

        Args:
            predictor (detectron2.engine.DefaultPredictor): default predictor
            feature_maps (dict): original pyramid feature maps
            new_bbox_lesion (Boxes): transformed bbox of lesion

        Returns:
            torch.Tensor: tensor with the features of the bbox, 256 features of 7x7
        """
        features = [feature_maps[f] for f in predictor.model.roi_heads.box_in_features]
        box_features = predictor.model.roi_heads.box_pooler(features, [new_bbox_lesion])
        return box_features
        
    def extract_1024(self, image_rgb:np.array, bbox_lesion:np.array):
        """Extract the 1024 features from the image and bbox

        Args:
            image_rgb (np.array): image to extract the features from
            bbox_lesion (np.array): bbox of the lesion

        Returns:
            np.array: 1024 features
        """
        if self.feature_name != 'features_1024':
            self.feature_name = 'features_1024' # update feature name
        # we make the bbox match the format of the backbone
        new_bbox_lesion = self.prepare_bbox(bbox_lesion, self.predictor, image_rgb)

        # BACKBONE FF-FEATURES
        feature_maps = self.backbone_feature_extraction(self.predictor, image_rgb)

        # BBOX POOLER FEATURES
        box_features = self.bbox_pooler_feature_extraction(self.predictor, feature_maps, new_bbox_lesion)

        # HEAD FEATURES (1024)
        box_features_after_head = self.predictor.model.roi_heads.box_head(box_features)
        # send to cpu
        box_features_after_head = box_features_after_head.detach().cpu().numpy()[0]
        assert box_features_after_head.shape == (1024,)
        return box_features_after_head
    
    def features_to_csv(self, features:np.array, pat_num:str):
        """Transform the features to a csv file. The first column is the patient ID, the rest are the features

        Args:
            features (np.array): features to transform, as a 1D array
            pat_num (strorint): patient number

        Returns:
            pd.DataFrame: dataframe with the features
        """
        range_f = features.shape[0]
        df = pd.DataFrame(features).T
        df.insert(0, 'PatientID', pat_num)
        df.columns = ['PatientID'] + [i for i in range(1, range_f+1)]

        return df
    
    def update_main_df(self, df:pd.DataFrame):
        """Update the main dataframe with the new features

        Args:
            df (pd.DataFrame): dataframe with the features
        """
        self.main_df = pd.concat([self.main_df, df], ignore_index=True)

    def save_main_df(self, rad:str, time:str, save_path:Path=None):
        """Save the main dataframe to a csv file

        Args:
            rad (str): radiologist name
            time (str or int): time of the exam
        """
        assert self.main_df is not None, 'The main dataframe is empty'
        # warning if the main_df is not complete
        if len(self.main_df) < 33:
            print('WARNING: The main dataframe is not complete')
        
        self.main_df_path = repo_path / f'data/deep/features/{self.feature_name}/{rad}_{time}_features.csv' if save_path is None else save_path
        # make sure parent exists
        self.main_df_path.parent.mkdir(parents=True, exist_ok=True)
        self.main_df.to_csv(self.main_df_path, index=False)
        print(f'Main dataframe saved to {self.main_df_path}')
        # reset main_df
        self.main_df = None
