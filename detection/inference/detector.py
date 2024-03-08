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
from detectron2.structures import boxes, pairwise_iou
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

setup_logger()

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

    def start_metrics(self):
        """start from zero the metrics for the current test.
        """
        # metrics
        self.FN_count = 0 # overall False Negatives
        self.TP_FP_dataframe = None
        self.froc_dataframe = None

    def prepare_im_gt(self):
        """Prepares the current image and ground truth for detection.
        Attributes changed :
        - self.c_im_array
        - self.c_gt_bboxes (x1, y1, x2, y2 format)
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
        # remove axis labels
        for a in ax:
            a.axis('off')
        fig.tight_layout()
        plt.show()

    def compute_TP_FN_counts(self, show=False):
        """Computes the number of True Positives and False Negatives for the current image.
        
        Attributes changed :
        - self.FN_count: general False Negatives count
        - self.TP_FP_dataframe: general dataframe with the TP and FP information

        Returns:
        - used_preds: list of indices of the predictions used for the TP count, which are not considered for the FP
        """
        # compute all ious between GT and predictions
        gt_pred_ious = pairwise_iou(boxes.Boxes(self.c_gt_bboxes), self.c_output.pred_boxes).numpy()
        self.c_gt_pred_ious = gt_pred_ious
        if show:
            print(f'GT_num: {gt_pred_ious.shape[0]}, Pred_num: {gt_pred_ious.shape[1]}')
            print(gt_pred_ious)

        # active outputs
        TP_dataframe = None
        used_preds = []
        
        ### Compute the TP
        for gt_num, _ in enumerate(self.c_gt_bboxes): #<--- go through each GT sample, the actual coordinates are not used
            # Case:1 FN if no prediction has iou >= 0.3
            if np.all(gt_pred_ious[gt_num] < 0.3):
                self.FN_count += 1
                continue
            # Case:2 TP with only one prediction with iou >= 0.3
            elif np.sum(gt_pred_ious[gt_num] >= 0.3) == 1:
                # get the index of the prediction with iou >= 0.3
                pred_num = np.argmax(gt_pred_ious[gt_num])
                # get the score of the prediction
                score = self.c_output.scores[pred_num]
                # save in the dataframe
                TP_FP_gt_df = pd.DataFrame(
                    {'image_name': self.c_im_name,
                    'category': 'TP',
                    'iou': gt_pred_ious[gt_num, pred_num],
                    'score': score.item(),
                    },
                    index=[0]
                    )
                TP_dataframe = pd.concat([TP_dataframe, TP_FP_gt_df])
                # # remove the prediction from the matrix                   ## <- rmeoving from the matrix is a bug!
                # gt_pred_ious = np.delete(gt_pred_ious, pred_num, axis=1) ## this will make losing track of the actual position of the used predictions
                # save the position of the used prediction
                used_preds.append(pred_num)
                continue
            # Case:3 TP with multiple predictions with iou >= 0.3
            elif np.sum(gt_pred_ious[gt_num] >= 0.3) > 1:
                # get the indices of the predictions with iou >= 0.3
                pred_nums = np.where(gt_pred_ious[gt_num] >= 0.3)[0]
                # get the scores of the predictions
                scores = self.c_output.scores[pred_nums]
                # select the prediction with the highest score, usually the first, given the detectron2 output
                selected_pred_num = np.argmax(scores)
                # save in the dataframe
                TP_FP_gt_df = pd.DataFrame(
                    {'image_name': self.c_im_name,
                    'category': 'TP',
                    'iou': gt_pred_ious[gt_num, selected_pred_num],
                    'score': scores[np.argmax(scores)].item(),
                    },
                    index=[0]
                    )
                TP_dataframe = pd.concat([TP_dataframe, TP_FP_gt_df])
                # remove all the predictions from the matrix
                # gt_pred_ious = np.delete(gt_pred_ious, pred_nums, axis=1) # <---- same as above
                # save the position of the used predictions
                used_preds.extend(pred_nums.tolist())
        
        # after finishing all GT, we can concat this image dataframe to the global one, only if there are any TP
        if TP_dataframe is not None:
            self.TP_FP_dataframe = pd.concat([self.TP_FP_dataframe, TP_dataframe], ignore_index=True)

        # remove duplicates from the used_preds
        used_preds = list(set(used_preds))

        return used_preds
    
    def compute_FP_counts(self, used_preds):
        """Computes the number of False Positives for the current image.<br>
        It is the continuation of the compute_TP_FN_counts method and must go after the latter.

        Args:
            used_preds (list): list of indices of the predictions used for the TP count, which are not considered for the FP

        Attributes changed :
        - self.TP_FP_dataframe: general dataframe with the TP and FP information
        """
        FP_dataframe = None
        # count all FP cases
        c_preds = self.c_output.scores
        # case one: no predictions
        if len(c_preds)==0:
            # no predictions so nothing to do
            pass
        # case two: predictions available
        else:
            # remove the used predictions, if any
            c_preds = np.delete(c_preds, used_preds) if len(used_preds)>0 else c_preds
            c_gt_pred_ious = np.delete(self.c_gt_pred_ious, used_preds, axis=1) if len(used_preds)>0 else self.c_gt_pred_ious
            # count the remaining predictions as FP
            # if there are no remaining predictions, nothing to do
            if len(c_preds)>0:
                for pred_num in range(len(c_preds)): #<==Go through all the remaining predictions
                    # create the dataframe
                    c_FP_df = pd.DataFrame(
                                {'image_name': self.c_im_name,
                                'category': 'FP',
                                'iou': c_gt_pred_ious[:,pred_num].max(), # the maximum iou with any GT, info only
                                'score': c_preds[pred_num].item(),
                                },
                                index=[0]
                                )
                    FP_dataframe = pd.concat([FP_dataframe, c_FP_df], ignore_index=True)

        # concat FP_frame with the general dataframe if it is not NOne
        if FP_dataframe is not None:
            self.TP_FP_dataframe = pd.concat([self.TP_FP_dataframe, FP_dataframe], ignore_index=True)   

    def compute_FROC(self):
        """computes the TPR and FPpI for the current self.TP_FP_dataframe. This is done after the TP, FP and FN counts are done.

        Attributes changed :
        - self.froc_dataframe: general dataframe with the TPR and FPpI information
        
        Returns:
        - self.froc_dataframe: general dataframe with the TPR and FPpI information.
        """
        # order TP_FP dataframe by score
        TP_FP_df = self.TP_FP_dataframe.sort_values(by='score', inplace=False, ascending=False).reset_index(drop=True)
        # count number of TP
        TP_FN_count = (TP_FP_df['category'] == 'TP').sum() + self.FN_count

        # compute TPR iterating over the dataframe that is sorted by score
        for iter in range(len(TP_FP_df)):
            TP_FP_df.loc[iter, 'TPR'] = (TP_FP_df.loc[0:iter, 'category'] == 'TP').sum() / TP_FN_count
            TP_FP_df.loc[iter, 'FPpI'] = (TP_FP_df.loc[0:iter, 'category'] == 'FP').sum() / len(self.test_df['image_name'].unique())
        
        self.froc_dataframe = TP_FP_df

        return self.froc_dataframe