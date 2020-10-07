import tensorflow.keras as keras
import numpy as np
import librosa
import sounddevice as sd
from librosa.core import istft
import time
import gc

MODEL_PATH = "model_with_others.h5"
SR = 22050
dur = 2
#word1 = "/home/ak/Desktop/lightControl_v2/2secondFiles/test/isiklariYak/isiklari-yak-189.wav"
#word2 = "/home/ak/Desktop/lightControl_v2/2secondFiles/test/isiklariSondur/isiklari-sondur-128.wav"

class _Keyword_Spotting_Service:

    model = None
    _mappings = [
        "isiklariYak",
        "isiklariSondur",
        "others"
    ]

    _instance = None

    def predict(self, data):
        MFCCs = self.preprocess(data)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
        preditions = self.model.predict(MFCCs)
        predicted_index = np.argmax(preditions)
        predicted_word = self._mappings[predicted_index]

        return predicted_word

    def preprocess(self, data, num_mfcc=13, n_fft=2048, hop_length=512):
        #signal, sr = librosa.load(file_path)
        print(data)
        print(len(data))
        if len(data)>=SR:
            data = data[:SR]

        #MFCCs = librosa.feature.mfcc(data, SR, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        MFCCs = librosa.feature.mfcc(data, sr=22050, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        return MFCCs.T

def Keyword_Spotting_Service():
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance


while True:
    print('#######################################################LISTENING#######################################################')
    data = sd.rec(int(dur * SR), SR, channels=1, blocking='True')
    data = data.reshape(-1)
    #print(data.shape)
    kss = Keyword_Spotting_Service()
    keyword1 = kss.predict(data)
    #keyword2 = kss.predict(data)
    print(f"Predicted word: {keyword1}")
    time.sleep(2.5)
    gc.collect()