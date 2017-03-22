from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import collections
import re
import os

import extract_chroma from chroma_extraction

# feature collection return type
Dataset = collections.namedtuple( 'Dataset', ['data', 'target'] )

# Load training set
def load_from_chromas( path_to_training_clips ):
    ''' File logic, sort through a directory of
        files from the IRMAS set, handling each
        audio file as a datapoint
        perform extraction, assign target
    '''
    target, data = [], []
    for wavfile in os.listdir( path_to_training_clips ):
        if wavfile.endswith( ".wav" ):
            data.append( extract_chroma( wavfile ) )
            target = re.split( "[]", wavfile )[1]
            target.append( target )

    target = np.array( target, dtype=np.str )

    return Dataset( data=data, target=target )

# construct training set, test set

# build, fit, and evaluate model

# perform classifications with model
