# add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import cv2 as cv
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline
import torch

# create class for the generation of high resolution inpanted mammograms
class InpaintingGenerator:
    def __init__(self, saving_dir:Path) -> None:
        self.load_data()
        self.current_row = None
        self.current_image = None
        self.current_mask = None
        # saving settings
        self.saving_dir = saving_dir
        self.im_saving_dir = self.saving_dir / 'images'
        self.im_saving_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):        
        # data
        self.im_dir = repo_path / 'data/CDD-CESM/images/substracted'
        self.metadata = pd.read_csv(repo_path / 'data/CDD-CESM/metadata' / 'normal_cases.csv')

    def set_generator(self, model_dir=None):
        model_dir=repo_path / 'generation/inpainting/results/CEM-small_mass_split-1' if model_dir is None else model_dir
        # model_dir = 'runwayml/stable-diffusion-inpainting'
        self.pipe = DiffusionPipeline.from_pretrained(
            model_dir,
            safety_checker=None,
            torch_dtype=torch.float16
        ).to("cuda")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()

    def read_image(self):
        im_path = self.im_dir / (self.current_row['Image_name'] + '.jpg')
        self.im_path = im_path.with_name(im_path.name.replace(' ', '')) # remove space if necessary
        im = cv.imread(str(self.im_path), cv.IMREAD_GRAYSCALE)
        self.current_image = im
    
    def get_breast_mask(self):
        _, _, mask = self.get_normal_BBox(self.current_image) # obtain breast binary mask
        kernel = np.ones((5,5),np.uint8)
        self.current_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    def get_closeup(self):

        # sample from range of the image size
        width = 512
        height = 512
        while True:
            x1 = np.random.randint(0, self.current_image.shape[1])
            y1 = np.random.randint(0, self.current_image.shape[0])
            x2 = x1 + width
            y2 = y1 + height

            # check that the box limits are inside the image
            if x2 <= self.current_image.shape[1] and y2 <= self.current_image.shape[0]:
                break
        self.current_closeup_coords = (x1, y1, x2, y2)
        im_closup = self.current_image[y1:y2, x1:x2]
        self.current_mask_closup = self.current_mask[y1:y2, x1:x2]  
        self.current_im_closup = np.repeat(im_closup[:,:,None], 3, axis=2)

    def get_lesion_bbox(self):

        for _ in range(3): # try three times to find a bbox

            # compute area stats
            area_range, ratio_range = self.compute_area_ratio()
            # create bbox
            bbox_mask, bbox = self.create_lesion_bbox(xrange=(0, 512), yrange=(0, 512), area_range=area_range, ratio_range=ratio_range)

            # check bbox is fully inside the mask three times
            if np.all(self.current_mask_closup[bbox[1]:bbox[3],bbox[0]:bbox[2]]==255):
                self.current_bbox_mask = bbox_mask
                self.current_bbox = bbox
                return True
        
        return False # did not find a bbox

    def get_normal_BBox(self, image):
        """This function returns the mask of the breast, as well as the boudnig box that encopasses it.

        Args:
            image (np.array): image as array

        Returns:
            omidb.bbox, np.array: returns omidb.box and mask image as np.arrays
        """

        mask = cv.threshold(image, 0, 255, cv.THRESH_BINARY)[1]
        nb_components, output, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=4)
        sizes = stats[:, -1]
        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]
        img2 = np.zeros(output.shape,dtype=np.uint8)
        img2[output == max_label] = 255
        contours, _ = cv.findContours(img2,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
        cnt = contours[0]
        aux_im = img2
        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(aux_im,(x,y),(x+w,y+h),(255,0,0),5)
        out_bbox = (x, y, x+w, y+h)
        
        return out_bbox, img2, mask # returns bounding box and mask image.

    def compute_area_ratio(self):
        """Computes both area and ratio statistics needed for lesion mas selection

        Returns:
            tuple: area and ratio statistics
        """

        training_mask_path = repo_path  /'data/CDD-CESM/metadata/bboxes/split_1/train_set.csv'
        train_set = pd.read_csv(training_mask_path)

        mask_areas = []
        mask_ratios = []
        for bbox in train_set['bbox']:
            bbox = eval(bbox)
            if bbox[2] < 512 and bbox[3] < 512: # excluding large cases
                # compute mask area
                mask_areas.append(bbox[2]*bbox[3])
                mask_ratios.append(bbox[3]/bbox[2])


        # get area 25th and 75th percentiles and ratio mean and std
        q25, q75 = np.percentile(mask_areas, 25), np.percentile(mask_areas, 75)
        mean, std = np.mean(mask_ratios), np.std(mask_ratios)

        area_range = (q25, q75)
        ratio_range = (mean - std, mean + std)

        return area_range, ratio_range

    def create_lesion_bbox(self, xrange, yrange, area_range, ratio_range):
        """Creates a lesion bbox given the ranges of the image and the area and ratio statistics

        Args:
            xrange (tuple): x range of the image
            yrange (tuple): y range of the image
            area_range (tuple): area range of the lesion
            ratio_range (tuple): ratio range of the lesion

        Returns:
            np.array, tuple: returns bbox, (x1, y1, x2, y2)
        """
        while True:
            # sample x and y
            x1 = np.random.randint(xrange[0], xrange[1])
            y1 = np.random.randint(yrange[0], yrange[1])
            # sample area and ratio
            area = np.random.randint(area_range[0], area_range[1])
            ratio = np.random.uniform(ratio_range[0], ratio_range[1])

            # compute width and height
            width = int(np.sqrt(area*ratio))
            height = int(np.sqrt(area/ratio))

            # compute other corners
            x2 = x1 + width
            y2 = y1 + height

            # if all corners are inside the patch range
            if x2 < xrange[1] and y2 < yrange[1]:
                # create mask of the bbox
                bbox_mask = np.zeros((xrange[1], yrange[1]), dtype=np.uint8)
                bbox_mask[y1:y2,x1:x2] = 255

                break
        
        return bbox_mask, (x1, y1, x2, y2)
    
    def select_lesion_patch_and_bbox(self):
        """creates the patch and bbox for the current image, to use for inpainting
        Main attributes changed:
        - self.current_im_closup
        - self.current_closeup_coords
        - self.current_bbox_mask
        """
        # for the current image case
        self.read_image()
        self.get_breast_mask()
        found_bbox = False
        while not found_bbox:
            self.get_closeup()
            found_bbox = self.get_lesion_bbox()
        # define general bbox coordinates
        self.current_bbox_general_coord = (self.current_closeup_coords[0] + self.current_bbox[0], self.current_closeup_coords[1] + self.current_bbox[1], 
                      self.current_closeup_coords[0] + self.current_bbox[2], self.current_closeup_coords[1] + self.current_bbox[3])
            
    def show_current_patch_and_bbox(self):
        # show preposessing images
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(self.current_image, cmap='gray')
        axs[0].set_title('Normal case')
        axs[0].add_patch(plt.Rectangle((self.current_closeup_coords[0], self.current_closeup_coords[1]),
                                        self.current_closeup_coords[2]-self.current_closeup_coords[0],
                                        self.current_closeup_coords[3]-self.current_closeup_coords[1], fill=False, edgecolor='r', lw=2))
        axs[1].imshow(self.current_im_closup, cmap='gray')
        axs[1].set_title(f'Closeup of 512x512, all image is white? {np.all(self.current_mask_closup==255)}')
        axs[2].imshow(self.current_im_closup, cmap='gray')
        axs[2].imshow(self.current_bbox_mask, cmap='jet', alpha=0.4)
        axs[2].set_title('Generated bbox mask')
        fig.suptitle('Random selection of closeup and lesion bbox')
        plt.show()

    def synthesize_lesion(self, diffusion_steps=50, guidance_scale=2, seed = None):
        """Synthesizes lesion in closeup and in final image
        Main attributes created:
        - self.current_closup_lesion: np.array
        - self.current_inpainted
        """

        self.set_generator() if not hasattr(self, 'pipe') else None # define generator
        # extra HP
        prompt = "a mammogram with a lesion"
        negative_prompt = ""
        num_samples = 1
        
        # randomness
        if seed is not None:
            generator = torch.Generator(device="cuda")
            generator.manual_seed(seed)
        else:
            generator = None

        with torch.autocast("cuda"), torch.inference_mode():
            synth = self.pipe(
                prompt=prompt,
                image=Image.fromarray(self.current_im_closup),
                mask_image=self.current_bbox_mask,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_samples,
                num_inference_steps=diffusion_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512,
                generator=generator,
            ).images[0]
        # closeup
        synth_array = np.asarray(synth)
        self.current_closup_lesion = synth_array
        # whole image
        self.current_inpainted = self.current_image.copy()
        ## Attention: only one channel is being used: channel
        channel_num = 0
        self.current_inpainted[self.current_closeup_coords[1]:self.current_closeup_coords[3], self.current_closeup_coords[0]:self.current_closeup_coords[2]
            ] = synth_array[:,:,channel_num]
        
    def show_synthetic_closup(self):
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        axs[0,0].imshow(self.current_im_closup, cmap='gray')
        axs[0,0].set_title('Original image')
        axs[0,1].imshow(self.current_closup_lesion, cmap='gray')
        axs[0,1].set_title('Generated image')
        # show histogram
        axs[1,0].hist(self.current_im_closup.ravel(), bins=256, range=(0, 255), fc='k', ec='k')
        axs[1,0].set_title('Original image')
        axs[1,1].hist(self.current_closup_lesion.ravel(), bins=256, range=(0, 255), fc='k', ec='k')
        axs[1,1].set_title('Generated image')
        fig.suptitle('Synthetic lesion in closeup')
        plt.show()
        
    def show_synthetic_inpainting(self, figsize=(20, 15), histogram=False):
        _, axs = plt.subplots(1, 2, figsize=figsize)
        axs[0].imshow(self.current_image, cmap='gray')
        axs[0].set_title('Original image')
        axs[1].imshow(self.current_inpainted, cmap='gray')
        axs[1].set_title('Inpainted image')
        if histogram:
            _, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].hist(self.current_image.ravel(), bins=256, range=(0, 255), fc='k', ec='k')
            axs[0].set_title('Original image')
            axs[1].hist(self.current_inpainted.ravel(), bins=256, range=(0, 255), fc='k', ec='k')
            axs[1].set_title('Generated image')
        plt.show()
        
    def save_synthetic_inpainting(self):
        # save image using same name as original
        saving_path = self.im_saving_dir / self.im_path.name
        cv.imwrite(str(saving_path), self.current_inpainted)
        
    def start_metadata(self):
        self.df_synthetic = pd.DataFrame(
            columns=['patient_id', 'image_name', 'region_id', 'bbox', 'lesions_per_patient'])
        
    def save_metadata(self):
        # from current_bbox_general_coord
        (x, y, w, h) = (self.current_bbox_general_coord[0], self.current_bbox_general_coord[1], 
                        self.current_bbox_general_coord[2]-self.current_bbox_general_coord[0], 
                        self.current_bbox_general_coord[3]-self.current_bbox_general_coord[1])
        current_df_synthetic = pd.DataFrame(
            {
                'patient_id': self.current_row['Patient_ID'],
                'image_name': self.current_row['Image_name'],
                'region_id': 1,
                'bbox': f'({x}, {y}, {w}, {h})',
                'lesions_per_patient': 1
            }, index=[0])
        
        # concat with general df_synthetic
        self.df_synthetic = pd.concat([self.df_synthetic, current_df_synthetic], axis=0)

number_samples = 415            

def main():
    inpainter = InpaintingGenerator(saving_dir= repo_path / 'generation/inpainting/data/inpainted_normal_cases')
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
