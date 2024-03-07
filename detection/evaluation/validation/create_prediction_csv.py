# add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

from detection.inference.detector import lesion_detector

def main():

    #### configuration: (ediatable)
    split_name = 'super_reduced_few_shots'
    ###

    split_type_dir = repo_path / Path(f'detection/training/results/{split_name}')
    # model inputs
    config_file = repo_path / 'detection/training/config_files/fine_tuning_CEM.yaml'
    min_score = 0.1 # minimum score threshold to keep the prediction

    # validation data inputs
    im_dir = repo_path / 'data/CDD-CESM/images/substracted' # images directory (can contain other not only test)
    metadata_path = repo_path / 'data/CDD-CESM/metadata/bboxes/split_1/val_set.csv' # val metadata (only val)

    model_type_dirs = [f for f in split_type_dir.iterdir() if f.is_dir()]
    
    for model_type_dir in model_type_dirs:
    
        # csv saving with FROC info
        saving_dir = repo_path / 'detection/evaluation/data/validation' / model_type_dir.parent.name / model_type_dir.name
        saving_dir.mkdir(parents=True, exist_ok=True)

        # collect all possible model steps
        step_list = list((repo_path / model_type_dir).rglob('*.pth')) # get all files with ending .pth
        step_list = [x for x in step_list if 'model_final' not in x.name] # remove model_final.pth
        step_list.sort(key=lambda x: int(x.name.split('_')[-1].split('.')[0]))

        for model_file in step_list:
            
            detector = lesion_detector(config_file, model_file, metadata_path, im_dir, min_score)
            detector.start_metrics()
            for im_name in detector.test_df['image_name'].unique()[0:]:
                detector.c_im_name = im_name
                detector.prepare_im_gt()
                detector.predict()
                # detector.show_c_predictions()
                # metrics computing
                used_preds = detector.compute_TP_FN_counts(show=False)
                detector.compute_FP_counts(used_preds)
            froc_info  = detector.compute_FROC()
            froc_info.to_csv(saving_dir / f'{model_file.stem}.csv', index=False)

if __name__ == "__main__":
    main()