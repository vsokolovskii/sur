import pandas as pd
import os

# find all directories ending with 'train'
def find_train_dirs(path):
    train_dirs = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir.endswith('train'):
                train_dirs.append(os.path.join(root, dir))
    return train_dirs

def find_test_dirs(path):
    test_dirs = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir.endswith('dev'):
                test_dirs.append(os.path.join(root, dir))
    return test_dirs

def data_description_create(dirs, mode='train'):
    # read the data description file
    df_non_png, df_tg_png = pd.DataFrame(), pd.DataFrame()
    df_non_wav, df_tg_wav = pd.DataFrame(), pd.DataFrame()
    for dir in dirs:
        if os.path.basename(dir).startswith('non'):
            # get absolute paths of png files in the directory
            png_files = [os.path.join(dir, file) for file in os.listdir(dir) if file.endswith('.png')]
            # get absolute paths of wav files in the directory
            wav_files = [os.path.join(dir, file) for file in os.listdir(dir) if file.endswith('.wav')]
            #pandas read list to dataframe
            df_non_png['file'] = png_files
            df_non_png['label'] = 0
            df_non_wav['file'] = wav_files
            df_non_wav['label'] = 0
        else:
            png_files = [os.path.join(dir, file) for file in os.listdir(dir) if file.endswith('.png')]
            # get absolute paths of wav files in the directory
            wav_files = [os.path.join(dir, file) for file in os.listdir(dir) if file.endswith('.wav')]
            df_tg_png['file'] = png_files
            df_tg_png['label'] = 1
            df_tg_wav['file'] = wav_files
            df_tg_wav['label'] = 1

    df = pd.concat([df_non_png, df_tg_png], axis=0)
    df1 = pd.concat([df_non_wav, df_tg_wav], axis=0)
    df.to_csv(f'pngs-{mode}.csv', index=False)
    df1.to_csv(f'wavs-{mode}.csv', index=False)

if __name__ == '__main__':
    train_dirs = find_train_dirs(os.getcwd())
    data_description_create(train_dirs, 'train')
    data_description_create(find_test_dirs(os.getcwd()), 'test')
