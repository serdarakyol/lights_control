import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import time

fs = 22050
seconds = 2
count = 511
for i in range(1):
    print("#############################################listening#############################################")
    myrecord = sd.rec(int(fs*seconds), samplerate=fs, channels=1)
    sd.wait()
    #time.sleep(1)
    write('./dataset/others/other-' + str(count) + '.wav',fs,myrecord)
    print("\n"*3,' '*42, 'File Saved','\n',' '*41,'other-' + str(count) + '.wav', "\n"*3)
    count+=1