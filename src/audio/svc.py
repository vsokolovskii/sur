import librosa
import numpy as np
import soundfile as sf
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.mixture import BayesianGaussianMixture
import pandas as pd
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os
import math
import pickle
import glob
import argparse

class VoiceActivityDetector:

    def __init__(self):
        self.step = 160
        self.buffer_size = 160 
        self.buffer = np.array([],dtype=np.int16)
        self.out_buffer = np.array([],dtype=np.int16)
        self.n = 0
        self.VADthd = 0.
        self.VADn = 0.
        self.silence_counter = 0

    def vad(self, _frame):
        frame = np.array(_frame) ** 2.
        result = True
        threshold = 0.08 # adaptive threshold
        thd = np.min(frame) + np.ptp(frame) * threshold
        self.VADthd = (self.VADn * self.VADthd + thd) / float(self.VADn + 1.)
        self.VADn += 1.

        if np.mean(frame) <= self.VADthd:
            self.silence_counter += 1
        else:
            self.silence_counter = 0

        if self.silence_counter > 20:
            result = False
        return result

    def add_samples(self, data):
		# Push new audio samples into the buffer
        self.buffer = np.append(self.buffer, data)
        result = len(self.buffer) >= self.buffer_size
        return result

    
    def get_frame(self):
        window = self.buffer[:self.buffer_size]
        self.buffer = self.buffer[self.step:]
        return window
		
    def process(self, data):
        if self.add_samples(data):
            while len(self.buffer) >= self.buffer_size:
                window = self.get_frame() # Framing
                if self.vad(window):  # speech frame
                    self.out_buffer = np.append(self.out_buffer, window)
        return self.out_buffer



# Function to extract MFCCs from an audio file
def extract_mfccs(file_path):
    audio, sample_rate = librosa.load(file_path, sr=16000, mono=True)
    y = VoiceActivityDetector().process(audio)
    mfccs = librosa.feature.mfcc(y=y, sr=sample_rate)
    return mfccs

def train():
    # load data from csv
    df_train = pd.read_csv('../wav-train.csv')
    df_test = pd.read_csv('../wav-dev.csv')

    # get file paths and labels
    file_paths = df_train['file'].tolist()
    labels = df_train['label'].tolist()
    file_paths_test = df_test['file'].tolist()
    labels_test = df_test['label'].tolist()

    # Extract MFCCs
    features = [extract_mfccs(file_path) for file_path in file_paths]
    test_features = [extract_mfccs(file_path) for file_path in file_paths_test]

    # Aggregate MFCCs for each recording
    X_train = np.array([np.hstack([np.mean(feat, axis=1), np.std(feat, axis=1)]) for feat in features])
    y_train = np.array(labels)
    X_test = np.array([np.hstack([np.mean(feat, axis=1), np.std(feat, axis=1)]) for feat in test_features])
    y_test = np.array(labels_test)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the SVC
    svc = SVC(kernel='linear', probability=True)
    svc.fit(X_train, y_train)
    # store the model
    import pickle
    filename = 'svc_audio.sav'
    pickle.dump(svc, open(filename, 'wb'))

    # Evaluate the SVC
    y_pred = svc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of SVC system (audio) on dev set:", accuracy)


def inference():
    filename = 'svc_audio.sav'
    svc = pickle.load(open(filename, 'rb'))

    file_paths_test = glob.glob("../../dataset/eval/*.wav")

    basename_test = [os.path.basename(file_path_test).split('.')[0] for file_path_test in file_paths_test]
    # Extract MFCCs
    test_features = [extract_mfccs(file_path) for file_path in file_paths_test]
    # Aggregate MFCCs for each recording
    X_test = np.array([np.hstack([np.mean(feat, axis=1), np.std(feat, axis=1)]) for feat in test_features])
    # Standardize features
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)
    # Evaluate the SVC and print the probabilities of each class
    y_pred = svc.predict(X_test)
    probs =  svc.predict_proba(X_test)

    with open("audio_results.csv", "w") as f:
        for file, pred, prob in zip(basename_test, y_pred, probs):
            f.write(f"{file}; {int(pred)}")
            for p in prob:
                f.write(f"; {math.log(p, 10)}")
            f.write("\n")



if __name__ == '__main__':
    # parse the argments
    parser = argparse.ArgumentParser(description='Audio classification')
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--inference', action='store_true', help='run inference')


    args = parser.parse_args()
    if args.train:
        train()
    if args.inference:
        inference()
