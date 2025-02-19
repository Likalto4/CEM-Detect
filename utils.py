#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import cv2 as cv
from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt
import pandas as pd
import ast
import numpy as np


class dataset_CDD_CESM:
    """Class to load the CDD-CESM dataset
    """
    def __init__(self, mode:str=None) -> None:
        # useful items
        self.alias = {'low-energy':'DM', 'substracted':'CESM'}
        # paths
        self.im_dir = repo_path / 'data/CDD-CESM/images'
        self.metadata_path = repo_path / 'data/CDD-CESM/metadata/Radiology_manual_annotations.xlsx'
        self.annotations_path = repo_path / 'data/CDD-CESM/metadata/Radiology_hand_drawn_segmentations_v2.csv'
        # read metadata
        self.metadata = pd.read_excel(self.metadata_path, sheet_name='all')
        self.annotations = pd.read_csv(self.annotations_path, converters={'region_shape_attributes':ast.literal_eval}) # read region_shape_attributes as a dictionary
        # # work with special mode only
        if mode is not None:
            assert mode in self.alias.keys(), f'Invalid mode. Use one of {self.alias.keys()}'
            self.metadata = self.metadata[self.metadata['Type'] == self.alias[mode]].reset_index(drop=True)
            mode_str_name = '_CM_' if mode=='substracted' else '_DM_'
            self.annotations = self.annotations[self.annotations['#filename'].str.contains(mode_str_name)].reset_index(drop=True)
            self.mode = mode
        else:
            self.mode = None
        # attributes
        self.patient_ids = self.metadata['Patient_ID'].unique()
        
    def redefine_metadata(self, metadata:pd.DataFrame):
        """redefine the metadata of the object

        Args:
            metadata (pd.DataFrame): new metadata
        """
        self.metadata = metadata
        self.patient_ids = self.metadata['Patient_ID'].unique()

    def __repr__(self) -> str:
        return f'CDD-CESM dataset with {len(self.patient_ids)} patients\nTotal images: {len(self.metadata)}'
    
    def get_images_paths(self, mode:str= None, metadata:pd.DataFrame = None):
        """given the metadata given all paths will be returned

        Args:
            mode (str, optional): dual energy protocol modality. Defaults to 'low-energy'or'substracted'.
            metadata (pd.DataFrame, optional): if not given, object metadata is used. Defaults to None.

        Raises:
            ValueError: if mode is not valid

        Returns:
            pd.Series: series with the paths of the images
        """

        if metadata is None: # if no input is given, use the class attribute
            metadata = self.metadata
        
        if mode is not None:
            if mode not in self.alias.keys():
                raise ValueError(f'Invalid mode. Use one of {self.alias.keys()}')
            metadata = metadata[metadata.Type == self.alias[mode]].reset_index(drop=True)
        else:
            mode = self.mode
        
        dm_images = str(self.im_dir) + f'/{mode}/' + metadata.Image_name + '.jpg'
        # ensure no space in the paths
        dm_images = dm_images.str.replace(' ', '')

        return dm_images

class patient_CDD(dataset_CDD_CESM):
    def __init__(self, patient_id:int, dataset = dataset_CDD_CESM()) -> None:
        super().__init__()
        self.patient_id = patient_id
        self.metadata = dataset.metadata[dataset.metadata.Patient_ID == patient_id].reset_index(drop=True)
        self.mode = dataset.mode
        # individual instantaneus attributes
        self.row_counter = -1
        self.image_lat = None
        self.image_view = None
        self.image_mode = None
        self.image_findings = None
        self.image_pathology = None
        self.image_tags = None

    def __repr__(self) -> str:
        return f'Patient {self.patient_id} with {len(self.metadata)} images'
    
    def set_image(self, step_num:int = 1 ,show_status:bool=True , mode:str =None, view:str = None, laterality:str = None):
        """set image as attributes of the object

        Args:
            step_num (int, optional): The number of row advances. Defaults to 1.
            show_status (bool, optional): _description_. Defaults to True.
            mode (str, optional): _description_. Defaults to None.
            view (str, optional): _description_. Defaults to None.
            laterality (str, optional): _description_. Defaults to None.
        """
        assert len(self.metadata) > 0, 'No images found for this patient in this metadata'
        self.row_counter += step_num
        # access the row accordint to row_counter
        im_row = self.metadata.iloc[self.row_counter]
        # set attributes of current image
        self.image_mode = 'low-energy' if im_row['Type'] == 'DM' else 'substracted'
        self.image_view = im_row['View']
        self.image_lat = im_row['Side']
        self.image_findings = im_row['Findings']
        self.image_pathology = im_row['Pathology Classification/ Follow up']
        self.image_tags = im_row['Tags']
        self.image_metadata = pd.DataFrame(im_row).T
        self.image_path = self.get_path()
        self.image_annotations = self.annotations[self.annotations['#filename'] == self.image_path.name].reset_index(drop=True)
        self.image_num_annotations = len(self.image_annotations)

        if show_status:
            print(f'Image {self.row_counter+1} of {len(self.metadata)}')

        # stop when reaching end of metadata
        if self.row_counter == len(self.metadata)-1:
            self.row_counter = -1
            print("End of patient's images, reseting counter to 0") if show_status else None

    def get_path(self):
        """get path of defined image

        Returns:
            str: path of the image, defined by get_images_paths
        """
        paths_series = self.get_images_paths(self.image_mode, self.image_metadata)
        if len(paths_series) > 1:
            raise ValueError('More than one image found')
        elif len(paths_series) == 0:
            raise FileNotFoundError('No image found')
            
        
        return Path(paths_series[0])
       
    def get_array(self, flip:bool = False, plot:bool = False):
        """get the array of the set image previously defined

        Args:
            flip (bool, optional): if to flip the right lateral images ot the left. Defaults to False.
            plot (bool, optional): plot or not to see the image fast. Defaults to False.

        Raises:
            FileNotFoundError: if the image is not found

        Returns:
            np.array: image as a numpy array
        """
        # check if path exists
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f'Image not found at {self.image_path}')
        
        # read the image
        image = cv.imread(str(self.image_path))
        if flip and self.image_lat == 'R':
            image = cv.flip(image, 1)
        
        if plot:
            plt.imshow(image)
            plt.axis('off')
            plt.show()
        
        return image
    
    def ellipse_reader(self, annot:dict):
        """read the coordinates of a ellipse-like ROI

        Args:
            annot (dict): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if annot['name']=='ellipse':
            center = (annot['cx'], annot['cy'])
            semiaxes = (annot['rx'], annot['ry'])
        elif annot['name']=='circle':
            center = (annot['cx'], annot['cy'])
            semiaxes = (annot['r'], annot['r'])
        else:
            raise ValueError('wring reader')

        return center, semiaxes
    
    def polygon_reader(self, annot:dict):
        """read the coordinates of a polygon-like ROI

        Args:
            annot (dict): annotations

        Raises:
            ValueError: if the reader is not correct
        """
        if annot['name'] not in ['polygon', 'polyline']:
            raise ValueError('wrong reader')
        vertices = annot['all_points_x'], annot['all_points_y']
        vertices = np.array(vertices).T

        return vertices
    
    def point_reader(self, annot:dict):
        """read the coordinates of a point-like ROI

        Args:
            annot (dict): annotations

        Raises:
            ValueError: if the reader is not correct
        """
        if annot['name']!='point':
            raise ValueError('wrong reader')
        center = (annot['cx'], annot['cy'])

        return center
    
    def plot_annotations(self, figsize=(14,7)):
        image = self.get_array(flip=False, plot=False)
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.imshow(image)
        ax.set_title(f'Pat_{self.image_path.stem}, findings: {self.image_findings}\nTags: {self.image_tags}, {self.image_num_annotations} regions\n \
        {self.image_metadata["Pathology Classification/ Follow up"].values[0]}')
        for region_num in range(self.image_num_annotations):
            dic_ex = self.image_annotations[self.image_annotations.region_id==region_num].region_shape_attributes.values[0]
            # plot annotation
            if dic_ex['name'] in ['ellipse','circle']:
                center, axes = self.ellipse_reader(dic_ex)
                ax.add_patch(Ellipse(center, 2*axes[0], 2*axes[1], fill=False, edgecolor='r'))
            elif dic_ex['name']=='polygon':
                vertices = self.polygon_reader(dic_ex)
                ax.plot(vertices[:,0], vertices[:,1], 'r')
            elif dic_ex['name']=='point':
                center = self.point_reader(dic_ex)
                ax.scatter(center[0], center[1], c='r', s=10)
            fig.tight_layout()
        plt.show()

    def patient_image_combinations(self, metadata:pd.DataFrame=None):
        if metadata is None:
            metadata = self.metadata
        side_array = metadata.Side.to_numpy()
        view_array = metadata.View.to_numpy()
        type_array = metadata.Type.to_numpy()
        # map type to the alias
        alias_inv = {v: k for k, v in self.alias.items()}
        type_array = np.array([alias_inv[i] for i in type_array])

        # put all arrays in one
        all_combinations = np.vstack((type_array, view_array, side_array)).T

        return all_combinations
    