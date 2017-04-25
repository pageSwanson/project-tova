''' https://www.tensorflow.org/get_started/tflearn for reference on neural network development '''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import collections
import os
import sys
import random

from extract_features import extract_features

# enable more extensive logging
tf.logging.set_verbosity(tf.logging.INFO)

# feature collection return type
Dataset = collections.namedtuple( 'Dataset', ['data', 'label'] )

def load_dataset( classes, path_to_set, fraction ):
    ''' Sort through a directory of files, handling each file as a datapoint
        Perform extraction ( using extract_features ), assign target ( label )

        Parameters

            path_to_set : str
                The path to the dataset directory

            fraction : float
                Fraction of files to use for this training round

        Returns

            Dataset : collections.namedtuple
                A tuple containing the data set and the corresponding targets

    '''

    # collect a list of all files in the training set
    wavfiles = []
    for i, directory in enumerate( os.listdir( path_to_set ) ):
        wavfiles = wavfiles + [ ( directory, name ) for name in os.listdir( path_to_set + '/' + directory ) if name.endswith( ".wav" ) ]

    # sample a random subset of data 
    if 0 < fraction and fraction < 1:
        file_limit = int( round( fraction * len( wavfiles ) ) )
        wavfiles = random.sample( wavfiles, file_limit )

    else:
        if fraction == 1:
            file_limit = len( wavfiles )

        else:
            return None # don't proceed without a fraction

    data = np.zeros( ( file_limit, 36 ), dtype=np.float32 )
    label = np.zeros( file_limit, dtype=np.int32 )

    for file_n, ( directory, name ) in enumerate( wavfiles ):
        data[ file_n, : ] = extract_features( path_to_set + '/' + directory + '/' + name )
        label[ file_n ] = classes[ directory ]

    return Dataset( data=data, label=label )

def use_network( usage, path_to_data, fraction=1 ):
    ''' Use the network and perform training or classification on a single file

        Parameters

            usage : str
                Specify intent for the neural network

                options are -t ( train ), -c ( classify )
                The training option includes testing and an evaluation

            path_to_data : str
                The file path to the dataset, or wavfile you want to classify

            fraction : float
                Fraction of training set to use (0, 1)

    '''

    # predefine class labels
    classes = dict( zip( [ 'vio', 'tru', 'pia', 'org', 'flu', 'cel' ], [ 0, 1, 2, 3, 4, 5 ] ) )

    fraction = float( fraction )

    # information on layer decisions can be found here
    # http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
    #
    # Used to determine number of neurons for the input layer, one for each feature ( 36 )
    feature_columns = [ tf.contrib.layers.real_valued_column( "", dimension=36 ) ]

    # output layers determined here, single layer, one neuron for each class label ( 6 )
    # size of hidden layer ( using rule of thumb found at link above, sqrt( in * out ) )
    # continued work : experiment with additional neurons
    classifier = tf.contrib.learn.DNNClassifier( feature_columns=feature_columns,
                                                 hidden_units=[ 16 ], # another layer marginally improves response, but loss accumulates quickly
                                                 n_classes=len( classes ),
                                                 model_dir="../model/voice_model",
                                                 config=tf.contrib.learn.RunConfig( save_checkpoints_secs=5 ) ) # monitor

    if usage == '-t':

        print( "\n___________________________" )
        print( "training on a {} percent set".format( fraction * 100 ) )
        print( "producing fixed validation set first..\n" )

        stats = []

        testing_set = load_dataset( classes, path_to_data + "/Testing", fraction )

        iterations = 1 if fraction == 1 else 6
        
        for itr in range(0, iterations): 
            training_set = load_dataset( classes, path_to_data + "/Training", fraction )

            def get_train_inputs():
                ''' construct a training batch using a specified fractional amount
                '''

                x = tf.constant(training_set.data)
                y = tf.constant(training_set.label)

                print( "training data shape,", x.get_shape().as_list() )
                print( "training data label,", y.get_shape().as_list() )

                return x, y

            def get_test_inputs():
                ''' construct a testing batch
                '''
                
                x = tf.constant( testing_set.data )
                y = tf.constant( testing_set.label )

                print( "validation data shape,", x.shape )
                print( "validation label shape,", y.shape )

                return x, y

            # https://www.tensorflow.org/get_started/monitors if desired

            print()

            # train model, specifying back prop. iterations (steps)
            classifier.fit( input_fn=get_train_inputs, steps=2000 )
            classifier.fit( input_fn=get_train_inputs, steps=2000 )
            classifier.fit( input_fn=get_train_inputs, steps=2000 )
            classifier.fit( input_fn=get_train_inputs, steps=2000 )

            results = classifier.evaluate( input_fn=get_test_inputs, steps=1 )
            stats.append( ( results[ 'accuracy' ], results[ 'loss' ] ) )
            
            print("\n_______ itr {} with accuracy, loss thus far... {} _______".format( itr, stats ) )
            print("\n{}\n".format( results ) ) 

    elif usage == '-c':

        def get_new_samples():
            return np.array( extract_features( path_to_data ), dtype=np.float32 )

        # classify, restoring model from previous training point
        predictions = list( classifier.predict( input_fn=get_new_samples ) )

        # print label with voice name (instrument) and index
        voice_predictions = []
        for result in predictions:
            for voice, label in classes.items():
                if result == label:
                    voice_predictions.append( ( voice, result ) )

        print( "Class Predictions on new samples:     {}\n".format( voice_predictions ) )

if __name__ == "__main__":
    try:
        use_network( sys.argv[1], sys.argv[2], sys.argv[3] )
    except IndexError:
        print("You must provide a usage option ( -t, -c ), file or directory path, and fraction of set (float).")
