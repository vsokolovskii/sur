import pandas as pd
import os
import glob

def data_description_create(dir_path, mode='train'):
    # prepare the data frames
    df_png = {'file': [], 'label': []}
    df_wav = {'file': [], 'label': []}

    png_files = glob.glob(os.path.join(dir_path, '**/*.png'), recursive=True)
    wav_files = glob.glob(os.path.join(dir_path, '**/*.wav'), recursive=True)

    for file in png_files:
        label = file.split('/')[-2]
        df_png['file'].append(file)
        df_png['label'].append(label)

    for file in wav_files:
        label = file.split('/')[-2]
        df_wav['file'].append(file)
        df_wav['label'].append(label)

    
    df_png = pd.DataFrame(df_png)
    df_wav = pd.DataFrame(df_wav)
    # save the data frames
    if mode == 'train':
        df_png.to_csv(os.path.join(os.getcwd(), 'src', 'pngs-train.csv'), index=False)
        df_wav.to_csv(os.path.join(os.getcwd(), 'src', 'wav-train.csv'), index=False)
    elif mode == 'dev':
        df_png.to_csv(os.path.join(os.getcwd(), 'src', 'pngs-dev.csv'), index=False)
        df_wav.to_csv(os.path.join(os.getcwd(), 'src', 'wav-dev.csv'), index=False)
    
    

if __name__ == '__main__':
    dataset_dir = os.path.join(os.getcwd(), 'dataset')
    train_dir = os.path.join(dataset_dir, 'train')
    dev_dir = os.path.join(dataset_dir, 'dev')
    
    if not os.path.exists(train_dir) or not os.path.exists(dev_dir):
        raise Exception('Dataset directories do not exist')
    
    data_description_create(train_dir, mode='train')
    data_description_create(dev_dir, mode='dev')

