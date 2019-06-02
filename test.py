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
import matplotlib.pylab as plt 

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

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
    return X, y, features

def read_model(file_name):
    '''
    '''
    data = pickle.load(open(file_name, 'rb' ))
    clfr = data['clfr']
    fs_on = data['feature_selection']
    features = data['features']
    return clfr, fs_on, features


def build_parser():
    '''
    build the parser 
    '''
    parser = argparse.ArgumentParser(
        description=("Test a cyberthreat model.\n"))
    parser.add_argument("-i", "--input", 
        help="where testing data are stored",
        required=True)
    parser.add_argument("-o", "--output", 
        help="plot file to save",
        required=True)
    parser.add_argument("-m", "--model",
        help="model file to use ",
        required=True)
    return parser

def main(args): 
    # read in the data and get the feature subset  
    X, y, features = read_file(args.input) 
    #y[y==1] = -1  # change the labels to be compatible with the isolation forest (change in the future) 
    #y[y==1] = 1  # change the labels to be compatible with the isolation forest (change in the future) 
    attack_start = np.where(y==-1)[0][0]
    #print(attack_start)

    # read the model 
    clfr, fs_on, features = read_model(args.model)
    
    if fs_on: 
        X = X[:, features]

    yhat = clfr.predict(X)
    ahat = clfr.decision_function(X)

    detection_error = len(np.where(y != yhat)[0])*1.0/len(y)

    plt.figure()
    plt.plot(ahat, color='red', label='A-score')
    plt.plot([attack_start, attack_start], [-1, 1], color='black', label='Attack Start')
    #plt.plot(yhat, color='blue', label='Prediction')
    plt.legend()
    plt.xlabel('Samples Processed')
    plt.title('MLABA Classification Results (Error='+str(100*detection_error)+')')
    #plt.show()
    plt.savefig(args.output + '.pdf', format='pdf')

    all_output = np.stack((y, yhat, ahat), axis=1)
    np.savetxt(args.output + '.csv', all_output, delimiter=',')

    print('Done')
    return None


if __name__ == '__main__': 
    parser = build_parser()
    args = parser.parse_args()
    main(args)