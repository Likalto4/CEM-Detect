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
    """Class to handle the detection of lesions in images.
    The user must define:
    - the path to the configuration file
    - the path to the model file
    - the path to the inference metadata file
    - the path to the directory containing the images to be tested
    - the minimu threshold score for detection
    """
    def __init__(self, cfg_path, model_path, metadata_path:Path, im_dir:Path, thresh_score=0.5) -> None:
        assert model_path.exists(), f"Model file not found in {model_path}"
        self.cfg = get_cfg()
        self.cfg.merge_from_file(str(cfg_path))
        self.cfg.MODEL.WEIGHTS = str(model_path)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh_score
        self.predictor = DefaultPredictor(self.cfg)
        # input data
        self.im_dir = im_dir
        self.test_df = pd.read_csv(metadata_path)
        # define image format from the content of im_dir
        self.im_format = next(self.im_dir.glob('*')).suffix
        # currents
        self.c_im_name = None

    def prepare_im_gt(self):
        """Prepares the current image and ground truth for detection.
        Attributes changed :
        - self.c_im_array
        - self.c_gt_bboxes
        """
        # prepare image
        im_path = self.im_dir / f'{self.c_im_name}{self.im_format}'
        self.c_im_array = cv.imread(str(im_path))
        # prepare ground truth
        im_bboxes = self.test_df[self.test_df['image_name']==self.c_im_name] # filter bboxes for this image
        bboxes_info = [eval(bbox) for bbox in im_bboxes['bbox']] # get all regions bboxes
        self.c_gt_bboxes = [[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]] for bbox in bboxes_info] # convert to x1, y1, x2, y2 format

    def predict(self, img_path):
        im = cv.imread(img_path)
        outputs = self.predictor(im)
        return outputs

    def evaluate(self):
        evaluator = COCOEvaluator(self.cfg.DATASETS.TEST[0], self.cfg, False, output_dir="./output/")
        val_loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])
        inference_on_dataset(self.predictor.model, val_loader, evaluator)