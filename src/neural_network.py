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
    for directory in os.listdir( path_to_set ):
        file_limit = round( fraction * len( [ name for name in os.listdir( '.' ) if os.path.isfile( name ) ] ) )
        current_file = 0
        for wavfile in os.listdir( path_to_set + '/' + directory ):
            wavfile = os.listdir( path_to_set + '/' + directory )
            if wavfile.endswith( ".wav" ):
                data.append( extract_features( path_to_set + '/' + directory + '/' + wavfile ) )
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
    if usage == "-t":
        # construct a training and testing set
        # build, fit, and evaluate model 
        training_set = load_dataset( path_to_data + "/Training" )
        testing_set = load_dataset( path_to_data + "/Testing" )

        # information on layer decisions can be found here
        # http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
        #
        # Specify that all features have real-value data
        # Used to determine number of neurons for the input layer, one for each feature ( 36 )
        feature_columns = [ tf.contrib.layers.real_valued_column( "", dimension=36 ) ]
        #
        # output layers determined here, single layer, one neuron for each class label ( 6 )
        # size of hidden layer, based on mean of input and output neurons ( 21 ) 
        classifier = tf.contrib.learn.DNNClassifier( feature_columns=feature_columns,
                                                     hidden_units=[ 21 ],
                                                     n_classes=6,
                                                     model_dir="./model/phrase_model" )

        # def get_inputs( data_set ):
        #     feature_cols = { k: tf.constant( data_set[ k ].values ) for k in FEATURES } # dictionary of tensors
        #     labels = tf.constant( data_set[ LABEL ].values ) # representing targets
        #     return feature_cols, labels

	def get_train_inputs():
	    # construct training data correctly and return in the form of a tensor
	    data = tf.constant(training_set.data)
	    target = tf.constant(training_set.target)

	    return data, target

        # Fit to the model, specifying how many steps to train
        # step is 2000 for the time being, this is nearly arbitrary 
        classifier.fit( input_fn=get_train_inputs(), steps=2000 )

        # If you want to track training progress, you can use a tensor flow monitor

        # Define the test inputs
	def get_test_inputs():
	    data = tf.constant(testing_set.data)
	    target = tf.constant(testing_set.target)

	    return data, target

        accuracy_score = classifier.evaluate( input_fn=get_test_inputs(), steps=2000 )["accuracy"]
        
        print("Test Accuracy: " + accuracy_score)

    elif usage == "-c":
        # Specify that all features have real-value data
        feature_columns = [ tf.contrib.layers.real_valued_column( "", dimension=36 ) ]
        # define model from directory ( requires that model exists prior )
        classifier = tf.contrib.learn.DNNClassifier( feature_columns=feature_columns,
                                                     hidden_units=[ 21 ],
                                                     n_classes=6,
                                                     model_dir="./model/phrase_model" )

        def get_inputs( data_set ):
            feature_cols = { k: tf.constant( data_set[ k ].values ) for k in FEATURES }
            labels = tf.constant( data_set[ LABEL ].values )
            return feature_columns, labels

        def new_samples():
            return np.array( extract_features( path_to_data ),
                             dtype=np.float32 )

        # perform classification with model
        # in this case, the data comes from a single wav file
        predictions = list( classifier.predict( input_fn=lambda : get_inputs( new_samples() ) ) )

        print("New Samples, Class Predictions:     {}\n".format( predictions ))

if __name__ == "__main__":
    try:
        use_network( sys.argv[1], sys.argv[2] )
    except IndexError:
        print("You must provide a usage option ( -t, -c ) and a file or directory path.")
