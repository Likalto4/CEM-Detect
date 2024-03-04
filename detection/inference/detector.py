# add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import cv2 as cv
import pandas as pd
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import setup_logger
setup_logger()


class lesion_detector:
    def __init__(self, cfg_path, model_path, thresh_score=0.5) -> None:
        assert model_path.exists(), f"Model file not found in {model_path}"
        self.cfg = get_cfg()
        self.cfg.merge_from_file(str(cfg_path))
        self.cfg.MODEL.WEIGHTS = str(model_path)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh_score
        self.predictor = DefaultPredictor(self.cfg)

        # input data
        metadata_path = repo_path / 'data/CDD-CESM/metadata/bboxes/split_1/test_set.csv'
        self.im_dir = repo_path / 'data/CDD-CESM/images/substracted'
        self.test_df = pd.read_csv(metadata_path)

    def predict(self, img_path):
        im = cv.imread(img_path)
        outputs = self.predictor(im)
        return outputs

    def evaluate(self):
        evaluator = COCOEvaluator(self.cfg.DATASETS.TEST[0], self.cfg, False, output_dir="./output/")
        val_loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])
        inference_on_dataset(self.predictor.model, val_loader, evaluator)