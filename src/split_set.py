import os
import sys
import random

def main( path_to_set ):
    ''' Sort through training set and create a new test directory
            A random 20 percent selection of the set
            Creating an 80 / 20 split of the set

    '''

    # collect a list of all files in the training set, from every sub directory
    filenames = []
    for directory in os.listdir( path_to_set + "/Training" ):
        filenames = filenames + [ ( directory, name ) for name in os.listdir( path_to_set + "/Training/" + directory ) if name.endswith( ".wav" ) ]

    # define a limit to sample based upon the specified fraction. Here it's 20 percent
    file_limit = int( round( .2 * len( filenames ) ) )

    if file_limit == 0:
        file_limit = 1

    print "Total files in set:", len( filenames )

    filenames = random.sample( filenames, file_limit ) # sample fraction of total set
    print "Number of files moved to test set:", file_limit

    for directory, name in filenames:
        os.rename( path_to_set + "/Training/" + directory + '/' + name, path_to_set + "/Testing/" + directory + '/' + name ) 
        
if __name__ == "__main__":
    try:
        main( sys.argv[1] )
    except IndexError:
        print("Provide the path for the dataset directory that you would like to split.")
