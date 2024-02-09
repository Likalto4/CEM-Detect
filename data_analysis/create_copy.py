#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

from PIL import Image
import numpy as np
from tqdm import tqdm

from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
)

def copy_from_dir(src_dir, dst_dir, preprocess):
    """
    Copy images from src_dir to dst_dir, applying the preprocess transformation
    """
    # get paths of only jpg files
    img_paths = [x for x in src_dir.glob('*.jpg') if x.is_file()]
    for img_path in tqdm(img_paths, desc=f'Copying {src_dir.name} to {dst_dir.name}'):
        img = Image.open(img_path)
        img = preprocess(img)
        img.save(dst_dir / img_path.name)

def main():

    # lets cut the CDD dataset into 512x512 images

    # origin dir
    low_energy_dir = repo_path / 'data/CDD-CESM/images/low-energy'
    substracted_dir = repo_path / 'data/CDD-CESM/images/substracted' 

    # destination dir
    low_energy_512_dir = repo_path / 'data/CDD-CESM_512' / 'low-energy'
    low_energy_512_dir.mkdir(parents=True, exist_ok=True)
    substracted_512_dir = repo_path / 'data/CDD-CESM_512' / 'substracted'
    substracted_512_dir.mkdir(parents=True, exist_ok=True)



    resolution = 512

    preprocess = Compose(
        [ # classic squared aspect-preserved centered image
            Resize(resolution),
            CenterCrop(resolution), 
        ]
    )

    copy_from_dir(low_energy_dir, low_energy_512_dir, preprocess)
    copy_from_dir(substracted_dir, substracted_512_dir, preprocess)

if __name__ == '__main__':
    main()