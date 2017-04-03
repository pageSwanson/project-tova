from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import collections
import os, argparse
import re

import extract_features from extract_features

# feature collection return type
Dataset = collections.namedtuple( 'Dataset', ['data', 'target'] )

def load_dataset( path_to_set ):
    ''' Sort through a directory of files, handling each file as a datapoint
        Perform extraction ( using extract_features ), assign target ( label )

        Parameters

        path_to_set : str
            The path to the dataset directory

        Returns

        Dataset : collections.namedtuple
            A tuple containing the data set and the corresponding targets

    '''
    target, data = [], []
    for wavfile in os.listdir( path_to_set ):
        if wavfile.endswith( ".wav" ):
            data.append( extract_features( wavfile ) )
            target = re.split( "[]", wavfile )[1]
            target.append( target )

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
        training_set = load_dataset( path_to_data + "/train" )
        testing_set = load_dataset( path_to_data + "/test" )
        
        # Specify that all features have real-value data
        feature_columns = [ tf.contrib.layers.real_valued_column( "", dimension=36 ) ]
        # Build neural network with layer unit specs
        # information on layer decisions can be found here
        # http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
        classifier = tf.contrib.learn.DNNClassifier( feature_columns=feature_columns,
                                                     hidden_units=[ ?, ?, ? ],
                                                     n_classes=5,
                                                     model_dir="./model/phrase_model" )

        def get_train_inputs():
            # construct training data correctly and return in the form of a tensor
            x = # tensor representing the data from the training set
            y = # representing targets
            return x, y

        # Fit to the model, specifying how many steps to train
        classifier.fit( input_fn=get_train_inputs, steps=? )

        # If you want to track training progress, you can use a tensor flow monitor

        def get_test_inputs():
            x = # tensor for test data
            y = # targets
            return x, y

        accuracy_score = classifier.evaluate( input_fn=get_test_inputs,
                                              steps=? )["accuracy"]

        print "Test Accuracy: {0:f}".format( accuracy_score )

    elif usage == "-c":
        # Specify that all features have real-value data
        feature_columns = [ tf.contrib.layers.real_valued_column( "", dimension=36 ) ]
        # define model from directory ( requires that model exists prior )
        classifier = tf.contrib.learn.DNNClassifier( feature_columns=feature_columns,
                                                     hidden_units=[ ?, ?, ? ],
                                                     n_classes=5,
                                                     model_dir="./model/phrase_model" )
        # perform classification with model
        # in this case, the data comes from a single wav file
        def new_samples():
            return np.array( extract_features( path_to_data ),
                             dtype=np.float32 )

        predictions = list( classifier.predict( input_fn=new_samples ) )

        print "New Samples, Class Predictions:     {}\n".format( predictions )

if __name__ == "__main__":
    try:
        use_network( sys.argv[1], sys.argv[2] )
    except IndexError:
        print "You must provide a usage option ( -t, -c ) and a file or directory path."
