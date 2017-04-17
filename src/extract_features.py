import sys
import numpy as np
import librosa as rosa
from scipy import io

def slice_chroma( chroma, num_slices ):
    '''Slice the chromagram and produce a sample set of frames
    
    Parameters
    
    chroma : np.ndarray [shape=(n_chroma, t)]
        Normalized energy for each chroma bin at each frame
    
    num_slices : int
        Number of evenly spaced slices to take from the chromagram
    
        26 slices with respect to the expected number of frames, based on
            Properties of audio (above)
            8192 point FFT
            8192 / 8 hop length
    
    Returns
    
    chroma_sliced : np.ndarray [shape=( n_chroma, len( frame_indices ) )]
        Select number of slices from the original chromagram

    '''

    interval = chroma.shape[1] / num_slices
    slices = [frame_i for frame_i in range( 0, chroma.shape[1] ) if frame_i % interval == 1]
    chroma_sliced = chroma[:, slices]

    print "data is rank:", chroma_sliced.ndim
    print "banks:", chroma_sliced.shape[0], "frames:", chroma_sliced.shape[1]

    return chroma_sliced

def extract_chroma( wavfile ):
    '''Produce a sliced version of a chromagram created using librosa ( http://librosa.github.io/librosa/ )
    
    Parameters
    
    wavfile : str
        Name of wav file to interpret
    
        From the IRMAS dataset ( http://www.mtg.upf.edu/download/datasets/irmas )
            3s audio time
            44.1 kHz sampling rate
            Single instrument
    
    Returns
    
    chroma_sliced : np.ndarray [shape=( n_chroma, len( slices ) )]
        Select number of slices from the original chromagram
    
    '''

    fs, y = io.wavfile.read( wavfile )

    # remove second channel if stereo
    if y.shape[1] > 1:
        y = y[:, 0]

    n_fft = 8192
    # shift by hop_length for each successive frame
    hop_length = n_fft / 8
    filterbank_params = {'n_chroma':36, 'ctroct':4.0, 'base_c':True}

    chroma = rosa.feature.chroma_stft( y=y, sr=fs, n_fft=n_fft, hop_length=hop_length, **filterbank_params )

    print "data is rank:", chroma.ndim
    print "banks:", chroma.shape[0], "frames:", chroma.shape[1]

    chroma_sliced = slice_chroma( chroma, num_slices=26 )

    return chroma_sliced

def extract_features( wavfile ):
    '''Generic function used to invoke the data processing, providing simple replacement
       of an extraction method

    '''
    return extract_chroma( wavfile )

if __name__ == "__main__":
    try:
        extract_features( sys.argv[1] )
    except IndexError:
        print "You must provide the name of a wav file to process."
