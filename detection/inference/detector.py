# add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

import numpy as np
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib import patches
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
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
        self.c_gt_bboxes = None
        self.c_output = None

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

    def predict(self):
        outputs = self.predictor(self.c_im_array)
        self.c_output = outputs["instances"].to("cpu")

    def show_c_predictions(self, figsize=(10,5)):
        fig,ax = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(self.c_im_name)
        # GT
        ax[0].imshow(self.c_im_array)
        ax[0].set_title('Ground truth')
        for bbox in self.c_gt_bboxes:
            rect = patches.Rectangle((bbox[0],bbox[1]), width = bbox[2]-bbox[0], height = bbox[3]-bbox[1],linewidth=1,edgecolor='r',facecolor='none')
            ax[0].add_patch(rect)
        # Pred
        ax[1].imshow(self.c_im_array)
        ax[1].set_title('Predicted')
        for i, bbox in enumerate(self.c_output.pred_boxes.tensor.numpy()):
            rect = patches.Rectangle((bbox[0],bbox[1]), width = bbox[2]-bbox[0], height = bbox[3]-bbox[1],linewidth=1,edgecolor='r',facecolor='none')
            ax[1].add_patch(rect)
            # add score
            ax[1].text(bbox[0],bbox[1],f'{self.c_output.scores[i]:.2f}', fontsize=6, color='w')
        plt.show()

##### post processing filtering functions (outside of the class for now)

def check_inside_contained(box_out, box_in):
    # check if box_in is inside box_out
    if (box_in[0] > box_out[0]) and (box_in[1] > box_out[1]) and (box_in[2] < box_out[2]) and (box_in[3] < box_out[3]):
        return True
    else:
        return False

def return_instance_to_remove(predicted_boxes, predicted_scores):
    for instance_i, bbox_i in enumerate(predicted_boxes):
        for instance_j, bbox_j in enumerate(predicted_boxes):
            if instance_i != instance_j:
                if check_inside_contained(box_out=bbox_i, box_in=bbox_j):
                    # print(f'box {instance_j} is inside box {instance_i}')
                    if predicted_scores[instance_i] > predicted_scores[instance_j]:
                        return instance_j
                    else:
                        return instance_i
                    
def pick_insideoutside_contrained(predicted_boxes, predicted_scores):
    while True:
        removable_idex = return_instance_to_remove(predicted_boxes, predicted_scores)
        if removable_idex is None:
            break
        # remove instance
        predicted_boxes = np.delete(predicted_boxes, removable_idex, axis=0)
        predicted_scores = np.delete(predicted_scores, removable_idex)

    return predicted_boxes, predicted_scores

def post_process_pred(out):

    predicted_boxes = out.pred_boxes.tensor.numpy()
    predicted_scores = out.scores.numpy()

    predicted_boxes, predicted_scores = pick_insideoutside_contrained(predicted_boxes, predicted_scores)

    # substitute the prediction with the new one
    new_out = out[:len(predicted_boxes)]
    new_out.pred_boxes = boxes.Boxes(predicted_boxes)
    new_out.scores = predicted_scores

    return new_out