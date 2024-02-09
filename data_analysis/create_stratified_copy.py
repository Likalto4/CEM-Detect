#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

from PIL import Image
from tqdm import tqdm

from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
)

from utils import dataset_CDD_CESM, patient_CDD

def main():
    # origin dir
    low_energy_dir = repo_path / 'data/CDD-CESM/images/low-energy'
    substracted_dir = repo_path / 'data/CDD-CESM/images/substracted' 
    # objective dir
    training_low = repo_path / 'data/CCDforBBDM/train/A'
    training_sub = repo_path / 'data/CCDforBBDM/train/B'
    training_low.mkdir(parents=True, exist_ok=True)
    training_sub.mkdir(parents=True, exist_ok=True)
    val_low = repo_path / 'data/CCDforBBDM/val/A'
    val_sub = repo_path / 'data/CCDforBBDM/val/B'
    val_low.mkdir(parents=True, exist_ok=True)
    val_sub.mkdir(parents=True, exist_ok=True)
    
    # transform
    resolution = 512
    preprocess = Compose(
        [ # classic squared aspect-preserved centered image
            Resize(resolution),
            CenterCrop(resolution), 
        ]
    )

    dataset = dataset_CDD_CESM()
    patient_ids = dataset.patient_ids
    training_patients = patient_ids[:int(len(patient_ids)*0.8)]
    # the rest is the validation set
    validation_patients = patient_ids[int(len(patient_ids)*0.8):]

    for pat_id in tqdm(training_patients):
        patient = patient_CDD(pat_id, dataset)
        while True:
            patient.set_image(show_status=False)
            im_path = patient.get_path()
            img = Image.open(im_path)
            img = preprocess(img)
            if 'low-energy' in str(im_path):
                dst_dir = training_low
            elif 'substracted' in str(im_path):
                dst_dir = training_sub
            img.save(dst_dir / im_path.name)
            
            if patient.row_counter==-1:
                break

    for pat_id in tqdm(validation_patients):
        patient = patient_CDD(pat_id, dataset)
        while True:
            patient.set_image(show_status=False)
            im_path = patient.get_path()
            img = Image.open(im_path)
            img = preprocess(img)
            if 'low-energy' in str(im_path):
                dst_dir = val_low
            elif 'substracted' in str(im_path):
                dst_dir = val_sub
            img.save(dst_dir / im_path.name)
            
            if patient.row_counter==-1:
                break

if __name__ == '__main__':
    main()