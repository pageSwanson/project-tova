import sys
import numpy as np
import librosa as rosa
import librosa.display
from scipy import io
import matplotlib.pyplot as plt

def main():
    try:
        # wav file is opened if provided
        fs, y = io.wavfile.read(sys.argv[1])
        # remove second channel
        if y.shape[1] > 1:
            y = y[:, 0]
    except IndexError:
        print "You need to provide the name of a wav file to read."
    else:
        n_fft = 4096 
        # shift by hop_length for each successive frame
        hop_length = n_fft * 1/4

        filterbankParams = {'n_chroma':36, 'ctroct':4.0, 'base_c':True}

        Cn = rosa.feature.chroma_stft(y=y, sr=fs, n_fft=n_fft, hop_length=hop_length, **filterbankParams)        
        print "banks:", Cn.shape[0], "frames:", Cn.shape[1]

        plt.figure()
        rosa.display.specshow(Cn, y_axis='chroma', x_axis='time')
        plt.colorbar()
        plt.title('Chromagram')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
        main()
