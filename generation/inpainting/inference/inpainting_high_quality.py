# add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

from tqdm import tqdm
from generation.inpainting.inpainter import InpaintingGenerator

number_samples = 312           

def main():
    inpainter = InpaintingGenerator(saving_dir= repo_path / 'generation/inpainting/data/split_1_wVal/medium_area_normal') #<-- change this
    inpainter.set_generator()
    inpainter.start_metadata()

    counter = tqdm(range(number_samples))
    # defining an example
    for _, inpainter.current_row in inpainter.metadata[:number_samples].iterrows():
        inpainter.select_lesion_patch_and_bbox()
        # inpainter.show_current_patch_and_bbox()
        inpainter.synthesize_lesion(diffusion_steps=50, guidance_scale=2, seed=0)
        inpainter.save_synthetic_inpainting()
        inpainter.save_metadata()
        #counter
        counter.update(1)
    counter.close()
    
    # save metadata
    inpainter.df_synthetic['image_name'] = inpainter.df_synthetic['image_name'].str.strip() # clean spaces
    inpainter.df_synthetic.to_csv(inpainter.saving_dir / 'metadata.csv', index=False)

if __name__ == '__main__':
    main()
