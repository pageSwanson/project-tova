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

            Tuple of
                Dataset : collections.namedtuple
                    A tuple containing the data set and the corresponding targets

                classes : dict
                    A dictionary mapping instrument labels to indices

    '''
    data = []
    label = []

    # collect a list of all files in the training set
    for i, directory in enumerate( os.listdir( path_to_set ) ):
        wavfiles = [ ( directory, name ) for name in os.listdir( path_to_set + directory ) if name.endswith( ".wav" ) ]

    # define a limit to sample based upon the specified fraction
    if 0 < fraction and fraction < 1:
        file_limit = int( round( fraction * len( wavfiles ) ) )
        wavfiles = random.sample( wavfiles, file_limit ) # sample fraction of total set
    elif fraction == 0:
        # don't proceed, that's weird
        return None

    for directory, name in wavfiles:
        data.append( extract_features( path_to_set + '/' + directory + '/' + name ) )
        print( "data shape with new example added", data.shape )
        label.append( classes[ directory ] )
        print( "label shape with new example added", label.shape )

    data = np.asarray( data, dtype=np.float32 )
    label = np.asarray( label, dtype=np.int32 )
    return Dataset( data=data, label=label )

def use_network( usage, path_to_data ):
    '''Use the network and perform training or classification on a single file

        Parameters

            usage : str
                Specify intent for the neural network
                
                options are -t ( train ), -c ( classify )
                The training option includes testing and an evaluation

            path_to_data : str
                The file path to the dataset, or wavfile you want to classify

    '''

    # define class labels for fitting
    classes = dict( zip( [ 'vio', 'tru', 'pia', 'org', 'flu', 'cel' ], [ 0, 1, 2, 3, 4, 5 ] ) )
    print(classes)

    # information on layer decisions can be found here
    # http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
    #
    # Specify that all features have real-value data
    # Used to determine number of neurons for the input layer, one for each feature ( 36 )
    feature_columns = [ tf.contrib.layers.real_valued_column( "", dimension=36 ) ]

    # output layers determined here, single layer, one neuron for each class label ( 6 )
    # size of hidden layer, based on mean of input and output neurons ( 20 ) 
    classifier = tf.contrib.learn.DNNClassifier( feature_columns=feature_columns,
                                                 hidden_units=[ 20 ],
                                                 n_classes=len( classes ),
                                                 model_dir="../model/voice_model" )

    if usage == '-t':
        # PERFORM TRAINING / TESTING ON THE MODEL

        training_set = load_dataset( classes, path_to_data + "/Training", .1 )

	def get_train_inputs():
            # construct a training batch using a specified fractional amount
            # return as a tensor

	    x = tf.constant(training_set.data)
	    y = tf.constant(training_set.label)

            print( "training data shape, data", x.get_shape().as_list() )
            print( "training data shape, label", y.get_shape().as_list() )

	    return x, y

        # Fit to the model, specifying how many steps to train
        # step is 2000 for the time being, this is nearly arbitrary 
        classifier.fit( input_fn=get_train_inputs, steps=1000 )
        classifier.fit( input_fn=get_train_inputs, steps=1000 )

        # If you want to track training progress, you can use a tensor flow monitor

        testing_set = load_dataset( classes, path_to_data + "/Testing", .1 )

        # Define the test inputs
	def get_test_inputs():

	    x = tf.constant(testing_set.data)
	    y = tf.constant(testing_set.label)

            print( "testing data shape, data", x.get_shape().as_list() )
            print( "testing data shape, label", y.get_shape().as_list() )

	    return x, y

        accuracy_score = classifier.evaluate( input_fn=get_test_inputs, steps=2000 )["accuracy"]
        
        print("Accuracy: {0:f}".format( accuracy_score ) )

    elif usage == '-c':
        # PERFORM CLASSIFICATION ON A SAMPLE

        def get_new_samples():
            return np.array( extract_features( path_to_data ),
                             dtype=np.float32 )

        # perform classification with model
        # in this case, the data comes from a set of 'real' samples
        predictions = list( classifier.predict( input_fn=get_new_samples ) )
        # print label with voice name (instrument)
        voice_predictions = []
        for result in predictions:
            for voice, label in classes.items():
                if result == label:
                    voice_predictions.append( ( voice, result ) )

        print( "New Samples, Class Predictions:     {}\n".format( voice_predictions ) )

if __name__ == "__main__":
    try:
        use_network( sys.argv[1], sys.argv[2] )
    except IndexError:
        print("You must provide a usage option ( -t, -c ) and a file or directory path.")
