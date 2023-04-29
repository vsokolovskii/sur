import pandas as pd
import os
import numpy as np


def fuse_hypermaram_search():
    image_pred = pd.read_csv('image/image_results.csv', header=None, sep=';')
    audio_pred = pd.read_csv('audio/audio_results.csv', header=None, sep=';')
    gt = pd.read_csv('wav-dev.csv')
    gt = {os.path.basename(gt.iloc[i, 0]).split('.')[0]: gt.iloc[i, 1] for i in range(len(gt))}

    alphas = {}

    # iterate ofrom 0 to 1 with step 0.1
    for alpha in np.arange(0, 1.05, 0.05):
        match = 0
        image_gt_match = 0
        audio_gt_match = 0
        fuse_match = 0
        for i in range(len(image_pred)):
            if image_pred.iloc[i, 0] == audio_pred.iloc[i, 0]:
                image_probs = np.array(image_pred.iloc[i, 3:].to_list())
                audio_probs = np.array(audio_pred.iloc[i, 2:].to_list())

                fused_probs = (image_probs*alpha + audio_probs*(1-alpha))
                fused_probs = normalize_log_probs(fused_probs)
                fused_pred = np.argmax(fused_probs) + 1

                if image_pred.iloc[i, 1] == audio_pred.iloc[i, 1] == gt[image_pred.iloc[i, 0]]:
                    match += 1
                else:
                    if gt[image_pred.iloc[i, 0]] == image_pred.iloc[i, 1]:
                        image_gt_match += 1
                    elif gt[image_pred.iloc[i, 0]] == audio_pred.iloc[i, 1]:
                        audio_gt_match += 1
                if fused_pred == gt[image_pred.iloc[i, 0]]:
                    fuse_match = fuse_match + 1
        alphas[alpha] = fuse_match/len(image_pred)

    best_alpha = max(alphas, key=alphas.get)

    print("The best alpha: ", best_alpha)

    print("Accuracy: ", alphas[best_alpha])

    return best_alpha


def log_sum_exp(log_probs):
    max_log_prob = np.max(log_probs)
    return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))

def normalize_log_probs(log_probs):
    log_sum = log_sum_exp(log_probs)
    return log_probs - log_sum



if __name__ == "__main__":
    best_alpha = 0.1
    # inference
    image_pred = pd.read_csv('image/image_results.csv', header=None, sep=';')
    audio_pred = pd.read_csv('audio/audio_results.csv', header=None, sep=';')
    match = 0

    # join the two dataframes by 0 column
    df = pd.merge(image_pred, audio_pred, on=0)
   
    with open('results.txt', 'w') as f:
        for i in range(len(df)):
            image_probs = np.array(df.loc[i, '2_x':'32_x'].to_list())
            audio_probs = np.array(df.loc[i, '2_y':].to_list())

            fused_probs = (image_probs*best_alpha + audio_probs*(1-best_alpha))
            # normalize the log probabilities so that they sum to 1
            fused_probs = normalize_log_probs(fused_probs)

            fused_pred = np.argmax(fused_probs) + 1
            
            if df.loc[i, '1_x'] == df.loc[i, '1_y']:
                match += 1
            f.write(f"{df.loc[i, 0]} {fused_pred}")
            
            for prob in fused_probs:
                f.write(f" {prob}")
            f.write("\n")

        print(f"Both systems agreed on {match}/{len(df)} predictions")
 