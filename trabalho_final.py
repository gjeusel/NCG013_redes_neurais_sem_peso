#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, re
reload(sys)
sys.setdefaultencoding('utf8') # problem with encoding

import argparse

import matplotlib
matplotlib.use("Qt4Agg") # enable plt.show() to display
import matplotlib.pyplot as plt

import logging as log
import errno # cf error raised bu os.makedirs

import pandas as pd
import seaborn as sns

import math
import itertools
import numpy as np

import struct

from sklearn.model_selection import train_test_split

# Classifications :
from sklearn.metrics import accuracy_score, roc_auc_score,\
                            precision_score, recall_score, \
                            confusion_matrix, classification_report

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier

from PyWANN.WiSARD import WiSARD


##############################################
########      Global variables :      ########

# Paths
script_path = os.path.abspath(sys.argv[0])
working_dir_path = os.path.dirname(script_path)
default_figures_dir = working_dir_path + "/figures"

# Cmap
from matplotlib.colors import ListedColormap
cmap_OrRd = ListedColormap(sns.color_palette("OrRd", 10).as_hex())
cmap_RdYlBu = ListedColormap(sns.color_palette("RdYlBu", 10).as_hex())

# Some colors
cblue_9 = sns.color_palette("RdBu", 10)[9] #_blue
cblue_8 = sns.color_palette("RdBu", 10)[8] # other_blue
cgrey_8 = sns.color_palette("RdGy", 10)[8] # _grey
cgrey_7 = sns.color_palette("RdGy", 10)[7] # _grey
corange_2 = sns.color_palette("RdYlGn", 10)[2]
corange_3 = sns.color_palette("RdYlGn", 10)[3]

color_green = sns.color_palette('BuGn', 10)[8]
color_blue = sns.color_palette("PuBu", 10)[8]
color_red = sns.color_palette("OrRd", 10)[8]
color_orange = sns.color_palette("YlOrBr", 10)[8]
##############################################

# Function to convert Dataframe in nice colored table :
def render_mpl_table(data, col_width=4.0, row_height=0.625, font_size=14,
                     row_colors=['w'],
                     edge_color='w',
                     bbox=[0, 0, 1, 1], # [left, bottom, width, height]
                     header_col_color=[color_blue],
                     header_row_color=[color_blue],
                     header_columns=0,
                     show_rowLabels=True,
                     ax=None, fig=None,
                     **kwargs):
    """ <pandas.DataFrame> to nice table. """
#{{{
    import six
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * \
                     np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    if show_rowLabels:
        mpl_table = ax.table(cellText=data.values, bbox=bbox,
                                rowLabels=data.index,
                                colLabels=data.columns,
                                **kwargs)
    else :
        mpl_table = ax.table(cellText=data.values, bbox=bbox,
                                colLabels=data.columns,
                                **kwargs)


    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 :
            cell.set_text_props(weight='bold', color='w')
            idx_color = k[1] % len(header_col_color)
            cell.set_facecolor( header_col_color[idx_color] )
        elif k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            idx_color = k[1] % len(header_row_color)
            cell.set_facecolor( header_row_color[idx_color] )
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])

    return fig, ax
#}}}


class DataTools:
    """
    - df_train : pandas.DataFrame of train set of numbers
        schema : ['jpg_path', 'number', 'im_data']

    - df_test : pandas.DataFrame of test set of numbers
        schema : ['jpg_path', 'number', 'im_data']

    - X_train, y_train
    - X_test, y_test
    - X_train_bin, X_test_bin

    - df_results : pandas.DataFrame of results obtained by classification
                ['Classifier', 'Characteristics', 'accuracy'])
    - df_wisard  : results obtain from wisard Classifier
                ['Bleaching', 'num_bits_addr', 'accuracy'])
    """

    def __init__(self, path_DF=working_dir_path+'/energy_efficiency.csv'):
#{{{
        print 'Reading csv file ...'
        self.df = pd.read_csv(path_DF)

        print 'Formatting datas ...'
        # Drop Orientation variable which is useless for classification
        self.df = self.df.drop('Orientation', axis=1) # axis 0 for rows and 1 for columns

        # Number of input variables :
        self.nX = self.df.shape[1] - 1

        # Sum of loads :
        self.df['Total Load'] = self.df['Heating Load'] + self.df['Cooling Load']
        self.df = self.df.drop(['Heating Load', 'Cooling Load'], axis=1)
        self.df = self.df.drop_duplicates()

        self.df_classif = self.to_classification_df()
        self.df_classif = self.df_classif.drop_duplicates()

        self.df_classif_bin = self.to_bin_df()

        # # constructing numpy arrays for train and test
        self.df_to_numpy_arrays()
        self.df_bin_to_numpy_arrays()

        # Initiate df_results :
        self.df_results = pd.DataFrame(columns=
                ['Classifier', 'Characteristics', 'accuracy'])

        self.df_wisard = pd.DataFrame(columns=
                ['Bleaching', 'num_bits_addr', 'accuracy'])
#}}}


    def plot_describe(self):
#{{{
        df_desc = self.df.describe()
        fig, axes = render_mpl_table(df_desc.iloc[:,0:self.nX], col_width=5.0, font_size=12)
        plt.savefig(default_figures_dir+'/csv_describe.png',
                    box_inches='tight')
        return fig, axes
#}}}

    def to_classification_df(self):
#{{{
        df_classif = self.df.iloc[:,0:self.nX]
        class_arr = []
        for e in self.df['Total Load']:
            if e < 25:
                class_arr.append('A')
            elif 25 <= e and e < 50:
                class_arr.append('B')
            elif 50 <= e and e < 75:
                class_arr.append('C')
            elif 75 <= e and e < 100:
                class_arr.append('D')

        df_classif['class'] = np.array(class_arr)
        return df_classif
#}}}

    def to_bin_df(self):
#{{{
        df_classif_bin = pd.DataFrame(data=None,
            columns=self.df_classif.columns, index=self.df_classif.index)

        n_reg_total = self.df_classif.shape[0]

        for i in range(n_reg_total):
            for j in range(self.nX):
                float32 = self.df_classif.iloc[i,j]
                float32_in_bin = float32_to_binary(float32)
                list_bin = bin_str_to_bin_list(float32_in_bin)
                df_classif_bin.iloc[i,j] = list_bin

        df_classif_bin['class'] = self.df_classif['class']

        return df_classif_bin
#}}}


    def df_to_numpy_arrays(self, percent_train=80.):
#{{{
        n_reg_total = self.df_classif.shape[0]
        n_reg_train = int(n_reg_total*80./100.)

        print "Copying data to numpy arrays X_train, y_train ..."
        self.X_train = self.df_classif.iloc[0:n_reg_train,
                                            0:self.nX].as_matrix()
        self.y_train = self.df_classif.iloc[0:n_reg_train,
                                            self.nX].as_matrix()

        print "Copying data to numpy arrays X_test, y_test ..."
        self.X_test = self.df_classif.iloc[n_reg_train+1:n_reg_total,
                                            0:self.nX].as_matrix()

        self.y_test = self.df_classif.iloc[n_reg_train+1:n_reg_total,
                                            self.nX].as_matrix()
#}}}

    def df_bin_to_numpy_arrays(self, test_size=0.2):
#{{{
        # flatten lists of bin :
        n_reg_total = self.df_classif_bin.shape[0]
        X_bin = np.zeros( (n_reg_total, (self.nX-1)*32), dtype=int)

        for i in range(n_reg_total):
            list_tmp = []
            for j in range(self.nX-1):
                list_tmp = list_tmp + self.df_classif_bin.iloc[i,j]
            X_bin[i,:] = np.array(list_tmp)

        # Constructing arrays randomly with train_test_split()
        self.X_bin_train, self.X_bin_test, self.y_bin_train, \
        self.y_bin_test = train_test_split(
            X_bin, self.df_classif_bin.iloc[:, self.nX].as_matrix(),
            test_size=test_size)
#}}}


    ##############################################
    # Classifiers :

    def GaussianProcessClassifier(self):
#{{{
        print "GaussianProcessClassifier :"
        gp = GaussianProcessClassifier()

        print "Training ..."
        gp.fit(self.X_train, self.y_train)

        print "Testing ..."
        y_predicted = gp.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_predicted)
        self.df_results.loc[self.df_results.shape[0]] = \
            ['GaussianProcess', '', accuracy]

        return y_predicted
#}}}

    def SupportVectorMachine(self):
#{{{
        print "Support Vector Machine :"
        clf = svm.SVC()

        print "Training ..."
        clf.fit(self.X_train, self.y_train)

        print "Testing ..."
        y_predicted = clf.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_predicted)
        self.df_results.loc[self.df_results.shape[0]] = \
            ['Support_Vector_Machine', '', accuracy]

        return y_predicted
#}}}

    def NearestNeighbors(self):
#{{{
        print "NearestNeighborsClassifier :"
        clf = neighbors.KNeighborsClassifier(n_neighbors=10)

        print "Training ..."
        clf.fit(self.X_train, self.y_train)

        print "Testing ..."
        y_predicted =clf.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_predicted)
        self.df_results.loc[self.df_results.shape[0]] = \
            ['NearestNeighbors', 'n_neighbors=10', accuracy]

        return y_predicted
#}}}

    def MultiLayerPerceptron(self, hidden_layer_sizes=(100)):
#{{{
        print "MultiLayerPerceptronClassifier :"
        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)

        charac = 'hidden_layer_sizes : ' + str(hidden_layer_sizes)

        print "Training ..."
        clf.fit(self.X_train, self.y_train)

        print "Testing ..."
        y_predicted =clf.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_predicted)
        self.df_results.loc[self.df_results.shape[0]] = \
            ['MultiLayerPerceptron', charac, accuracy]

        return y_predicted
#}}}

    def WiSARD(self, num_bits_addr=37, bleaching=False, **kwargs):
        print "WiSARD : num_bits_addr=" + str(num_bits_addr)
        clf = WiSARD(num_bits_addr=num_bits_addr, **kwargs)

        print "Training ..."
        clf.fit(self.X_bin_train, self.y_bin_train)

        print "Testing..."
        y_predicted = clf.predict(self.X_bin_test)

        accuracy = accuracy_score(self.y_bin_test, y_predicted)

        self.df_wisard.loc[self.df_wisard.shape[0]] = \
            [bleaching, num_bits_addr, accuracy]

        if bleaching:
            txt = "WiSARD with bleaching"
        else:
            txt = "WiSARD"
        self.df_results.loc[self.df_results.shape[0]] = \
            [txt, 'num_bits_addr = ' + str(num_bits_addr), accuracy]

def float32_to_binary(float32):
#{{{
    # Struct can provide us with the float packed into bytes. The '!' ensures that
    # it's in network byte order (big-endian) and the 'f' says that it should be
    # packed as a float. Alternatively, for double-precision, you could use 'd'.
    packed = struct.pack('!f', float32)
    # print 'Packed: %s' % repr(packed)

    # For each character in the returned string, we'll turn it into its corresponding
    # integer code point
    # [62, 163, 215, 10] = [ord(c) for c in '>\xa3\xd7\n']
    integers = [ord(c) for c in packed]
    # print 'Integers: %s' % integers

    # For each integer, we'll convert it to its binary representation.
    binaries = [bin(i) for i in integers]
    # print 'Binaries: %s' % binaries

    # Now strip off the '0b' from each of these
    stripped_binaries = [s.replace('0b', '') for s in binaries]
    # print 'Stripped: %s' % stripped_binaries

    # Pad each byte's binary representation's with 0's to make sure it has all 8 bits:
    #
    # ['00111110', '10100011', '11010111', '00001010']
    padded = [s.rjust(8, '0') for s in stripped_binaries]
    # print 'Padded: %s' % padded

    # At this point, we have each of the bytes for the network byte ordered float
    # in an array as binary strings. Now we just concatenate them to get the total
    # representation of the float:
    return ''.join(padded)
#}}}

def binary_to_float32(binary):
#{{{
    f = int(binary, 2) # base 2
    float32 = struct.unpack('f', struct.pack('I', f))[0]
    return float32
#}}}

def bin_str_to_bin_list(bin_str):
    bin_list = []
    for i in bin_str:
        bin_list.append(i)
    return bin_list

def get_sample(df):
    n_registers = df.shape[0]
    df_sample_head = df.head(3)
    df_sample_mid = df.iloc[n_registers/2 : n_registers/2 + 3,:]
    df_sample_tail = df.tail(3)
    df_sample = pd.concat([df_sample_head, df_sample_mid, df_sample_tail])
    return df_sample





def setup_argparser():
    """ Define and return the command argument parser. """
#{{{
    parser = argparse.ArgumentParser(description='''Building's Energy Efficiency Study.''')

    return parser
#}}}

def setup_paths(list_of_paths):
    """ Create defaults directories if needed. """
    for p in list_of_paths:
        try:
            os.makedirs(p)
        except OSError as exc: # Python >2.5
            if exc.errno != errno.EEXIST:
                raise

def main():

    parser = setup_argparser()

    try:
        args = parser.parse_args()

    except argparse.ArgumentError as exc:
        log.exception('Error parsing options.')
        parser.error(str(exc.message))
        raise

    setup_paths([default_figures_dir])

    dfs = DataTools()

    # fig, axes = render_mpl_table(get_sample(dfs.df))
    # fig, axes = dfs.plot_describe()

    dfs.GaussianProcessClassifier()
    dfs.SupportVectorMachine()
    # dfs.NearestNeighbors()
    # dfs.MultiLayerPerceptron()

    dfs.WiSARD()

    from IPython import embed; embed() # Enter Ipython


    plt.show() # interactive plot

if __name__ == '__main__':
    main()
