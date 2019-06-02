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
import warnings
import argparse
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')

def build_parser():
    '''
    build the parser 
    '''
    parser = argparse.ArgumentParser(
        description=("Preprocess the data.\n"))
    parser.add_argument("-i", "--input", 
        help="input file where selected are stored",
        required=True)
    parser.add_argument("-f", "--fs",
        help="number of features to select",
        required=False,
        type=int,  
        default=True)
    parser.add_argument("-p", "--trainpercent",
        help="How much data to train with. ",
        required=False,
        type=float,  
        default=0.95)
    return parser


def main(args): 
    # load the user inputs 
    percent_train = args.trainpercent
    file_path = args.input 
    top_k_features = args.fs

    # load the data set and print some information out about the data that we are looking at 
    # DATA FILE 
    #  - expanded_shalaka_logical_firefox.csv (2234431, 16555)
    #  - expanded_pratik.csv (1442460, 6696)
    df = pd.read_csv(file_path)
    print(df.keys()[1:])
    print('\n\n')
    print('# of features: ' + str(len(df.keys())-1))
    print('\n\n')
    print(df.head())
    print('\n\n')
    print('Normal Samples: ' + str(len(np.where(df['label'] == 'NORMAL')[0])))
    print('Attack Samples: ' + str(len(np.where(df['label'] == 'Attack_3a')[0])))
    print('\n\n')
    print(np.unique(df['label']))
    feature_names = df.keys()[1:]


    # copy the data to a new dataframe and relabel the data [NORMAL = 1, ATTACK = -1]
    df2 = df.copy()
    df2['label'][np.where(df2['label'] == 'NORMAL')[0]] = 1
    df2['label'][np.where(df2['label'] == 'Attack_3a')[0]] = -1

    # run feature selection 
    data = df2.values
    X = data[:, 1:]
    y = data[:, 0]

    good_samples = np.where(y==0)[0]
    bad_samples = np.where(y==1)[0]
    new_indx = np.concatenate((good_samples[:50000], bad_samples))
    Xnew = X[new_indx, :]
    ynew = y[new_indx]

    mi_scores = mutual_info_classif(Xnew, ynew, n_neighbors = 5)
    feature_ranks = np.argsort(mi_scores)[::-1][:top_k_features]
    print('Feature Ranks')
    p = 0
    for i in feature_ranks:
        print('(' + str(p) + '): ' + feature_names[i] + ' (' + str(mi_scores[i]) + ')')
        p += 1

    keep_features = feature_ranks[:top_k_features]

    # write a training and testing data file
    tr_data_path = file_path[:-4] + '_TRAIN.pkl'
    te_data_path = file_path[:-4] + '_TEST.pkl'
    tr_stop = int(np.floor(percent_train*len(df2)))

    # rearrange the data
    all_data = df2.values
    good = np.where(y==0)[0]
    malicious = np.where(y==1)[0]
    all_data_sorted = np.concatenate((all_data[good, :], all_data[malicious, :]), axis=0)

    data_tr = all_data_sorted[:tr_stop]
    X = data_tr[:, 1:]
    y = data_tr[:, 0]
    data = {'X': X, 'y': y, 'features': keep_features}
    outfile = open(tr_data_path, 'wb')
    pickle.dump(data, outfile)
    outfile.close()
    print('Training Samples: ' + str(len(y)))

    data_te = all_data_sorted[tr_stop:]
    X = data_te[:, 1:]
    y = data_te[:, 0]

    data = {'X': X, 'y': y, 'features': keep_features}
    outfile = open(te_data_path, 'wb')
    pickle.dump(data, outfile)
    outfile.close()
    print('Testing Samples: ' + str(len(y)))

    return None


if __name__ == '__main__': 
    parser = build_parser()
    args = parser.parse_args()
    main(args)

