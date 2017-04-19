from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import collections
import os, argparse
import sys
import re

from extract_features import extract_features

# feature collection return type
Dataset = collections.namedtuple( 'Dataset', ['data', 'target'] )

def load_dataset( path_to_set, fraction ):
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
    target, data = [], []

    # collect a list of all files in the training set
    for directory in os.listdir( path_to_set ):
        wavfiles = [ ( directory, name ) for name in os.listdir( path_to_set + '/' + directory ) if os.path.isfile( name ) and name.endswith( ".wav" ) ]

    # define a limit to sample based upon the specified fraction
    if 0 < fraction and fraction < 1:
        file_limit = round( fraction * len( wavfiles ) )
        wavfiles = random.sample( wavfiles, file_limit ) # sample fraction of total set

    for directory, name in wavfiles:
        data.append( extract_features( path_to_set + '/' + directory + '/' + name ) )
        target.append( directory )

    data = np.array( data, dtype=np.float32 )
    target = np.array( target, dtype=np.str )
    return Dataset( data=data, target=target )

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
                                                 n_classes=6,
                                                 model_dir="./model/voice_model" )

    if usage == '-t':
        # PERFORM TRAINING / TESTING ON THE MODEL

        training_set = load_dataset( path_to_data + "/Training", .2 )

	def get_train_inputs():
            # construct a training batch using a specified fractional amount
            # return as a tensor

	    data = tf.constant(training_set.data)
	    target = tf.constant(training_set.target)

	    return data, target

        # Fit to the model, specifying how many steps to train
        # step is 2000 for the time being, this is nearly arbitrary 
        classifier.fit( input_fn=get_train_inputs, steps=1000 )
        classifier.fit( input_fn=get_train_inputs, steps=1000 )

        # If you want to track training progress, you can use a tensor flow monitor

        testing_set = load_dataset( path_to_data + "/Testing", .2 )

        # Define the test inputs
	def get_test_inputs():

	    data = tf.constant(testing_set.data)
	    target = tf.constant(testing_set.target)

	    return data, target

        accuracy_score = classifier.evaluate( input_fn=get_test_inputs, steps=2000 )["accuracy"]
        
        print("Accuracy: {0:f}".format( accuracy_score ) )

    elif usage == "-c":
        # PERFORM CLASSIFICATION ON A SAMPLE

        def get_new_samples():
            return np.array( extract_features( path_to_data ),
                             dtype=np.float32 )

        # perform classification with model
        # in this case, the data comes from a set of 'real' samples
        predictions = list( classifier.predict( input_fn=get_new_samples ) )

        print("New Samples, Class Predictions:     {}\n".format( predictions ))

if __name__ == "__main__":
    try:
        use_network( sys.argv[1], sys.argv[2] )
    except IndexError:
        print("You must provide a usage option ( -t, -c ) and a file or directory path.")
