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

def AUFROC_computing(froc_info, FPpI_limit=1.0):
    """compute the FROC area under the curve.
    By thefault the AUC is computed up to 1 FPpI, if the limit is surpassed, the curve is cut at the limit.

    Args:
        froc_info (_type_): _description_
        FPpI_limit (float, optional): _description_. Defaults to 1.0.

    Returns:
        float: AUC value
    """ 
    # check if the FPpI limit is reached
    if froc_info['FPpI'].max() < FPpI_limit:
        # add a point to reach the limit, adding a row
        froc_info = pd.concat([froc_info, pd.DataFrame({'FPpI': [FPpI_limit], 'TPR': [froc_info['TPR'].iloc[-1]]})], ignore_index=True)
    # check if the FPpI limit is surpassed
    elif froc_info['FPpI'].max() > FPpI_limit:
        # remove points that surpass the limit
        froc_info = froc_info[froc_info['FPpI'] <= FPpI_limit]
        # add a point to reach the limit, adding a row
        froc_info = pd.concat([froc_info, pd.DataFrame({'FPpI': [FPpI_limit], 'TPR': [froc_info['TPR'].iloc[-1]]})], ignore_index=True)

    # compute the area under the curve using the trapezoidal rule
    AUC_value = np.trapz(froc_info['TPR'], x=froc_info['FPpI'])

    return AUC_value
    

def computing_sensitivity_at_FPpI(froc_info):
    # compute sensitivity at:
    sen_at = [1/2, 1, 2, 3]
    sensitivity_df = None
    for FPpI in sen_at:
        c_sen = np.interp(x=FPpI, xp=froc_info['FPpI'], fp=froc_info['TPR'])
        c_sen_df = pd.DataFrame({'FPpI': [FPpI], 'sensitivity': [c_sen]})
        sensitivity_df = pd.concat([sensitivity_df, c_sen_df], ignore_index=True)

    average_sen = sensitivity_df['sensitivity'].mean()
    return sensitivity_df, average_sen
    

def main():
    #### configuration: (ediatable)
    split_name = 'standard_split_wVal'
    ###
    split_type_dir = repo_path / Path(f'detection/training/results/{split_name}')
    model_type_names = [f.name for f in split_type_dir.iterdir() if f.is_dir()]
    
    for model_type_name in model_type_names:
        FROC_info_dir = repo_path / f'detection/evaluation/data/validation/{split_name}' / model_type_name
        
        # saving_AUFROCs and sensitivity
        AUFROC_dir = FROC_info_dir.parent / 'AUFROC'
        Sen_mean_dir = FROC_info_dir.parent / 'sensitivity'
        AUFROC_dir.mkdir(parents=True, exist_ok=True)
        Sen_mean_dir.mkdir(parents=True, exist_ok=True)
        
        # get all csv files
        csv_files = list(FROC_info_dir.rglob('*.csv'))

        AUFROC_df = None
        Sen_mean_df = None
        # example of model curve
        for csv_path in csv_files:

            model_name = csv_path.stem
            froc_info = pd.read_csv(csv_path)

            # compute metrics
            AUFROC_value = AUFROC_computing(froc_info)
            AUFROC_df = pd.concat([AUFROC_df, pd.DataFrame({'model_name': [model_name], 'metric': [AUFROC_value]})], ignore_index=True)
            # sensitivity
            _, sen_mean = computing_sensitivity_at_FPpI(froc_info)
            Sen_mean_df = pd.concat([Sen_mean_df, pd.DataFrame({'model_name': [model_name], 'metric': [sen_mean]})], ignore_index=True)
        # sort by metric
        AUFROC_df = AUFROC_df.sort_values(by='metric', ascending=False)
        Sen_mean_df = Sen_mean_df.sort_values(by='metric', ascending=False)
        
        # save
        AUFROC_df.to_csv(AUFROC_dir / f'{FROC_info_dir.name}.csv', index=False)
        Sen_mean_df.to_csv(Sen_mean_dir / f'{FROC_info_dir.name}.csv', index=False)

if __name__ == "__main__":
    main()