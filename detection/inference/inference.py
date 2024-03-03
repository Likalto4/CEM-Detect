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


def get_dicts(dataframe, im_dir):
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

def main():
    # configuration model
    config_file = repo_path / 'data/models/config_trained_R_101_30k.yaml'
    
    model_type = 'normal_fine-tuning' # original_DBT, fine_tuned
    if model_type == 'original_DBT': 
        model_file = repo_path / 'data/models/model_final_R_101_omidb_30k_dbt9k_f12_gray.pth' # original DBT weights
    else:
        model_file = repo_path / 'detection/training/data' / model_type / f'model_0014999.pth' # fine-tuned 2
    assert model_file.exists(), f"Model file not found in {model_file}"
    # special overrides
    # model_file = repo_path / 'detection/training/output/model_0009999.pth'
    
    # saving paths
    saving_name = 'fine_tuned'
    output_dir = repo_path / 'detection/inference/results' / saving_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # input data
    metadata_path = repo_path / 'data/CDD-CESM/metadata/bboxes/split_1/test_set.csv'
    im_dir = repo_path / 'data/CDD-CESM/images/substracted'
    test_dataframe = pd.read_csv(metadata_path)
    # print info
    print(f'Number of patients: {len(test_dataframe["patient_id"].unique())}')
    print(f'Number of lesions: {len(test_dataframe)}')
    print(f'Number of images: {len(test_dataframe["image_name"].unique())}')

    # register the dataset
    registration_name = "CEM_test"
    if registration_name in DatasetCatalog.list():
        DatasetCatalog.remove(registration_name)
        MetadataCatalog.remove(registration_name)
    DatasetCatalog.register(registration_name, lambda: get_dicts(test_dataframe, im_dir))
    MetadataCatalog.get(registration_name).set(thing_classes=["mass"])

    # model setting
    min_score = 0.01 # minimum score to keep prediction
    cfg = get_cfg()
    cfg.merge_from_file(str(config_file))
    cfg.MODEL.WEIGHTS = str(model_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min_score  # set the testing threshold for this model
    cfg.DATASETS.TEST = (registration_name, )
    # define predictor
    predictor = DefaultPredictor(cfg)

    # log info about the model
    print(f'The model min IoU score is: {min_score}')

    # define evaluator
    evaluator = COCOEvaluator(dataset_name=registration_name, distributed=False, output_dir=str(output_dir), allow_cached_coco=False)
    val_loader = build_detection_test_loader(cfg, registration_name)
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(metrics)

    # save csv
    pd.DataFrame(metrics).to_csv(output_dir / 'COCO_metrics.csv', index=True)

if __name__ == "__main__":
    main()