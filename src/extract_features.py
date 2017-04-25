import sys
import numpy as np
import librosa as rosa
from scipy import io

def slice_chroma( chroma, num_slices ):
    ''' Slice the chromagram and produce a set with the predominant moment of each note
    
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
        
            chroma_sliced : np.ndarray [shape=( n_chroma, 1 ) )]
                Take the average over every note, produce expected magnitude

    '''

    # slicing, without statistic, preserved for analysis
    #interval = chroma.shape[1] / num_slices
    #slices = [frame_i for frame_i in range( 0, chroma.shape[1] ) if frame_i % interval == 1]
    #chroma_sliced = chroma[:, slices]

    chroma_sliced = np.mean( chroma, axis=1 )
    chroma_sliced = chroma_sliced.reshape( 1, chroma_sliced.shape[ 0 ] )

    return chroma_sliced

def extract_chroma( wavfile ):
    ''' Produce a sliced version of a chromagram created using librosa ( http://librosa.github.io/librosa/ )
    
        Parameters
        
            wavfile : str
                Name of wav file to interpret
            
                From the IRMAS dataset ( http://www.mtg.upf.edu/download/datasets/irmas )
                    3s audio time
                    44.1 kHz sampling rate
                    Single instrument
        
        Returns
        
            chroma_sliced : np.ndarray [shape=( n_chroma, 1 ) )]
                Take the average over every note, produce expected magnitude

    '''

    fs, y = io.wavfile.read( wavfile )

    # remove second channel for stereo
    if y.shape[1] > 1:
        y = y[:, 0]

    # narrowband interpretation for accurate frequency identification
    n_fft = 8192
    # shift by hop_length for each successive frame
    hop_length = n_fft / 8
    filterbank_params = {'n_chroma':36, 'ctroct':4.0, 'base_c':True}

    chroma = rosa.feature.chroma_stft( y=y, sr=fs, n_fft=n_fft, hop_length=hop_length, **filterbank_params )

    chroma_sliced = slice_chroma( chroma, num_slices=26 )

    return chroma_sliced

def extract_features( wavfile ):
    ''' Generic function used to invoke the data processing, providing simple replacement of an extraction method '''

    return extract_chroma( wavfile )

if __name__ == "__main__":
    try:
        extract_features( sys.argv[1] )
    except IndexError:
        print "You must provide the name of a wav file to process."
