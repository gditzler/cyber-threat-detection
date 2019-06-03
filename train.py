#/usr/bin/env python 

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pickle
import argparse
import pickle
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.ensemble import IsolationForest 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

def read_file(file_name):
    '''
    read in the pickle file that will have a dictionary that has the variables X, y and 
    features, where feature were the features that were selected in the preprocessing 
    of the data in the Jupyter notebook 
    '''
    data = pickle.load(open(file_name, 'rb' ))
    X = data['X']
    y = data['y']
    features = data['features']
    scaler = data['scalemodel']
    return X, y, features, scaler

def build_parser():
    '''
    build the parser 
    '''
    parser = argparse.ArgumentParser(
        description=("Implementation of CyberThreatModels.\n"))
    parser.add_argument("-i", "--input", 
        help="input file where selected are stored",
        required=True)
    parser.add_argument("-m", "--model",
        help="model file to save ",
        required=True)
    parser.add_argument("-c", "--classifier",
        help="if or auto",
        required=False, 
        default='if')
    parser.add_argument("-f", "--fs",
        help="if or auto",
        required=False,
        type=bool,  
        default=True)
    parser.add_argument("-s", "--scale",
        help="scale the data?",
        required=False,
        type=bool,  
        default=False)
    parser.add_argument("-v", "--verbose",
        help="turn on verbose",
        action="store_true", 
        required=False)
    return parser

def train_autoencoder():
    '''
    TO DO: implement an autoencoder. 
    ''' 
    return None


def main(args):
    '''
    main program function that will read in the data, run feature selection if the user
    has requested to do so then train the detection model 
    '''
    # read in the data file 
    if args.verbose: 
        print('Loading the data file...\n\n')
    X, _, features, scaler = read_file(args.input)   # the _ is for the labels which are not used

    # scale the data if the user wants to do so (this flag for scaling should be set if 
    # the user is implementing the autoencoder. 
    #if args.scale: 
    #    X = scaler.transform(X)

    # if we are going to run feature selection then grab the features that were selected
    if args.fs:
        if args.verbose: 
            print('Reducing the feature set the the specified features...\n\n') 
        X = X[:, features]
    
    if args.verbose: 
        print('Training the classifier...\n\n')
    
    if args.classifier == 'if': 
        clfr = IsolationForest(n_estimators=100, 
                               max_samples='auto', 
                               contamination=0.1, 
                               max_features=1, 
                               bootstrap=True, 
                               random_state=None, 
                               verbose=0).fit(X)
    else:
        raise Exception('A detection model was specified that has not been implemented.')

    if args.verbose: 
        avg_score = np.mean(clfr.predict(X))
        print('The average IF score for normal training data is: ' + str(avg_score) + '\n\n')
     
    model = {'clfr': clfr, 'features': features, 'feature_selection': args.fs}
    print(model)
    outfile = open(args.model, 'wb')
    pickle.dump(model, outfile)
    outfile.close()

    if args.verbose: 
        print('The model has been saved to ' + args.model)

    return None 

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    main(args)
