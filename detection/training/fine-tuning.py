# add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None
sys.path.insert(0,str(repo_path / 'detr')) if str(repo_path / 'detr') not in sys.path else None

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import pairwise_iou, boxes
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.solver.build import maybe_add_gradient_clipping

from typing import Any, Dict, List, Set
import itertools
import random

import torch
import cv2 as cv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from detectron2.utils.logger import setup_logger
setup_logger()

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=None)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer
    
def get_CEM_dicts(dataframe, im_dir):
        """define a ddataset dictionary for detectron2

        Args:
            dataframe (pd.DataFrame): dataframe with the following columns:
            image_name, bbox, image_id, patient_id

        Returns:
            list: list of dictionaries with the following keys:
            file_name, image_id, height, width, annotations
        """
        dataset_dicts = []
        df = dataframe.copy()

        df['image_id'] = df['image_name'].astype('category').cat.codes


        for _, row in df.iterrows():
            
            record = {}
            
            filename = im_dir / (row["image_name"] + ".jpg")
            height, width = cv.imread(str(filename)).shape[:2]
            
            record["file_name"] = str(filename)
            record["image_id"] = row["image_id"]
            record["height"] = height
            record["width"] = width
            

            bbox_roi = eval(row["bbox"])   
            px = [bbox_roi[0], bbox_roi[0]+bbox_roi[2], bbox_roi[0]+bbox_roi[2], bbox_roi[0]]
            py = [bbox_roi[1], bbox_roi[1], bbox_roi[1]+bbox_roi[3], bbox_roi[1]+bbox_roi[3]]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            objs = []
            obj =  {
                    "bbox": [bbox_roi[0] , bbox_roi[1], bbox_roi[0]+bbox_roi[2], bbox_roi[1]+bbox_roi[3]],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                }
            objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

        return dataset_dicts

def register_dataset(registraiton_name:str):
    # input paths
    # csv_meta_path = repo_path / 'detection/preprocessing/data/combined_datasets/split_1/improved_augmentation/metadata.csv' # <---- change
    # csv_meta_path = repo_path / 'data/CDD-CESM/metadata/bboxes/split_1/train_set.csv'
    csv_meta_path = repo_path / 'generation/inpainting/data/split_1/improved_normal/metadata.csv'

    # im_dir = repo_path / 'detection/preprocessing/data/combined_datasets/split_1/improved_augmentation/images' # <---- change
    # im_dir = repo_path / 'data/CDD-CESM/images/substracted'
    im_dir = repo_path / 'generation/inpainting/data/split_1/improved_normal/images'
    
    # clean image_name column, no black spaces
    df = pd.read_csv(csv_meta_path)
    df['image_name'] = df['image_name'].str.strip()
    # register our dataset
    DatasetCatalog.clear()
    DatasetCatalog.register(registraiton_name, lambda: get_CEM_dicts(dataframe=df, im_dir=im_dir))
    MetadataCatalog.get(registraiton_name).set(thing_classes=["mass"])
    

def main():
    # parameters
    iter_num = 30000
    
    # dataset
    registraiton_name = "CEM_train"
    register_dataset(registraiton_name=registraiton_name)
    
    # training configuration and weights
    config_file = repo_path / 'detection/training/config_files/fine_tuning_CEM.yaml'
    model_file = repo_path / 'data/models/model_final_R_101_omidb_30k_dbt9k_f12_gray.pth'
    cfg = get_cfg()
    cfg.merge_from_file(str(config_file))
    cfg.MODEL.WEIGHTS = str(model_file)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = iter_num #10000    # 300 iterations is enough for a toy dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (mass).
    cfg.DATASETS.TRAIN = (registraiton_name,)
    cfg.DATASETS.TEST = ()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    trainer = DefaultTrainer(cfg) # or Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()

