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
default_figures_dir = working_dir_path + "/figures_trabalho_final"

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

    return fig, ax, mpl_table
#}}}


class DataTools_classif:
    """
    - df : pandas.DataFrame
        schema : ['Relative Compactness', 'Surface Area', 'Wall Area',
                  'Roof Area', 'Overall Height', 'Orientation',
                  'Glazing Area', 'Glazing Area Distribution',
                  'class']

    - df_bin : pandas.DataFrame, df but in list of binaries

    - X_train, X_test
    - y_train, y_test

    - X_bin_train, X_bin_test
    - y_bin_train, y_bin_test

    - df_results : pandas.DataFrame of results obtained by classification
                ['Classifier', 'Characteristics', 'accuracy'])
    - df_wisard  : results obtain from wisard Classifier
                ['Bleaching', 'num_bits_addr', 'accuracy'])
    """
#{{{

    def __init__(self, path_DF=working_dir_path+'/energy_efficiency.csv'):
#{{{
        print '---------- Initiating DataTools_classif instance ----------'
        print 'Reading csv file ...'
        self.df = pd.read_csv(path_DF)

        print 'Generating df ...'
        # Drop Orientation variable which is useless for classification
        self.df = self.df.drop('Orientation', axis=1) # axis 0 for rows and 1 for columns

        # Sum of loads :
        self.df['Total Load'] = self.df['Heating Load'] + self.df['Cooling Load']
        self.df = self.df.drop(['Heating Load', 'Cooling Load'], axis=1)
        self.df = self.df.drop_duplicates()

        # Number of input variables :
        self.nX = self.df.shape[1] - 1

        # Converting last column for classification purpose :
        self.df = self.to_classification_df()
        self.df = self.df.drop_duplicates()

        print 'Generating df_bin ...'
        self.df_bin = self.to_classif_bin_df()

        # Initiate df_results :
        self.df_results = pd.DataFrame(columns=
                ['Classifier', 'Characteristics', 'accuracy'])

        self.df_wisard = pd.DataFrame(columns=
                ['Bleaching', 'num_bits_addr', 'accuracy'])
#}}}


    ##############################################
    # functions used to construct DataFrames :

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

    def to_classif_bin_df(self):
#{{{
        df_bin = pd.DataFrame(data=None,
            columns=self.df.columns, index=self.df.index)

        n_reg_total = self.df.shape[0]

        for i in range(n_reg_total):
            for j in range(self.nX):
                float32 = self.df.iloc[i,j]
                float32_in_bin = float32_to_binary(float32)
                list_bin = bin_str_to_bin_list(float32_in_bin)
                df_bin.iloc[i,j] = list_bin

        df_bin['class'] = self.df['class']

        return df_bin
#}}}


    def df_to_numpy_arrays(self, test_size=0.1):
#{{{
        n_reg_total = self.df.shape[0]

        X_train, X_test, y_train, y_test = \
            train_test_split(self.df.iloc[:,0:self.nX],
                             self.df.iloc[:,self.nX],
                             test_size=test_size)

        return X_train, X_test, y_train, y_test
#}}}

    def df_bin_to_numpy_arrays(self, test_size=0.1):
#{{{
        # flatten lists of bin :
        n_reg_total = self.df_bin.shape[0]
        X_bin = np.zeros( (n_reg_total, (self.nX-1)*32), dtype=int)

        for i in range(n_reg_total):
            list_tmp = []
            for j in range(self.nX-1):
                list_tmp = list_tmp + self.df_bin.iloc[i,j]
            X_bin[i,:] = np.array(list_tmp)

        # Constructing arrays randomly with train_test_split()
        X_bin_train, X_bin_test, y_bin_train, y_bin_test = \
            train_test_split(
            X_bin, self.df_bin.iloc[:, self.nX].as_matrix(),
            test_size=test_size)

        return X_bin_train, X_bin_test, y_bin_train, y_bin_test
#}}}


    ##############################################
    # tools for vizualization :

    def plot_describe(self):
#{{{
        df_desc = self.df.describe()
        fig, axes = render_mpl_table(df_desc.iloc[:,0:self.nX], col_width=5.0, font_size=12)
        plt.savefig(default_figures_dir+'/csv_describe.png',
                    box_inches='tight')
        return fig, axes
#}}}



    ##############################################
    # Classifiers :

    def GaussianProcessClassifier(self, ncross_val=20):
#{{{
        print "GaussianProcessClassifier with ", ncross_val,\
              " times cross-validation :"

        accuracy = [None]*ncross_val
        for n in range(ncross_val):
            self.X_train, self.X_test, self.y_train, self.y_test = \
                self.df_to_numpy_arrays()

            gp = GaussianProcessClassifier()
            gp.fit(self.X_train, self.y_train)
            y_predicted = gp.predict(self.X_test)

            accuracy[n] = accuracy_score(self.y_test, y_predicted)

        self.df_results.loc[self.df_results.shape[0]] = \
            ['GaussianProcess', '', sum(accuracy)/float(len(accuracy))]

        return y_predicted
#}}}

    def SupportVectorMachine(self, ncross_val=20):
#{{{
        print "Support Vector Machine with ", ncross_val,\
              " times cross-validation :"

        accuracy = [None]*ncross_val
        for n in range(ncross_val):
            self.X_train, self.X_test, self.y_train, self.y_test = \
                self.df_to_numpy_arrays()

            clf = svm.SVC()
            clf.fit(self.X_train, self.y_train)
            y_predicted = clf.predict(self.X_test)

            accuracy[n] = accuracy_score(self.y_test, y_predicted)

        self.df_results.loc[self.df_results.shape[0]] = \
            ['Support_Vector_Machine', '',
             sum(accuracy)/float(len(accuracy))]

        return y_predicted
#}}}

    def NearestNeighbors(self, ncross_val=20, n_neighbors=4):
#{{{
        print "NearestNeighborsClassifier with ", ncross_val,\
              " times cross-validation :"

        accuracy = [None]*ncross_val
        for n in range(ncross_val):
            self.X_train, self.X_test, self.y_train, self.y_test = \
                self.df_to_numpy_arrays()

            clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
            clf.fit(self.X_train, self.y_train)
            y_predicted =clf.predict(self.X_test)

            accuracy[n] = accuracy_score(self.y_test, y_predicted)

        self.df_results.loc[self.df_results.shape[0]] = \
            ['NearestNeighbors', 'n_neighbors='+str(n_neighbors),
             sum(accuracy)/float(len(accuracy))]

        return y_predicted
#}}}

    def MultiLayerPerceptron(self, ncross_val=20, hidden_layer_sizes=(100)):
#{{{
        print "MultiLayerPerceptronClassifier with ", ncross_val,\
              " times cross-validation :"

        accuracy = [None]*ncross_val
        for n in range(ncross_val):
            self.X_train, self.X_test, self.y_train, self.y_test = \
                self.df_to_numpy_arrays()

            clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
            clf.fit(self.X_train, self.y_train)
            y_predicted =clf.predict(self.X_test)

            accuracy[n] = accuracy_score(self.y_test, y_predicted)

        charac = 'hidden_layer_sizes : ' + str(hidden_layer_sizes)
        self.df_results.loc[self.df_results.shape[0]] = \
            ['MultiLayerPerceptron', charac,
             sum(accuracy)/float(len(accuracy))]

        return y_predicted
#}}}

    def WiSARD_classify(self, ncross_val=20, num_bits_addr=50,
                        bleaching=False, **kwargs):
#{{{
        print "WiSARD : num_bits_addr=" + str(num_bits_addr), " and ", \
            ncross_val, " times cross-validation"

        accuracy = [None]*ncross_val
        for n in range(ncross_val):
            self.X_bin_train, self.X_bin_test, self.y_bin_train, \
                self.y_bin_test = self.df_bin_to_numpy_arrays()

            clf = WiSARD(num_bits_addr=num_bits_addr, **kwargs)
            clf.fit(self.X_bin_train, self.y_bin_train)
            y_predicted = clf.predict(self.X_bin_test)

            accuracy[n] = accuracy_score(self.y_bin_test, y_predicted)

        self.df_wisard.loc[self.df_wisard.shape[0]] = \
            [bleaching, num_bits_addr, sum(accuracy)/float(len(accuracy))]

        if bleaching:
            txt = "WiSARD with bleaching"
        else:
            txt = "WiSARD"
        self.df_results.loc[self.df_results.shape[0]] = \
            [txt, 'num_bits_addr = ' + str(num_bits_addr),
             sum(accuracy)/float(len(accuracy))]
#}}}

    def calling_them_all(self, ncross_val):
        self.GaussianProcessClassifier(ncross_val=ncross_val)
        self.SupportVectorMachine(ncross_val=ncross_val)
        self.NearestNeighbors(ncross_val=ncross_val)
        self.MultiLayerPerceptron(ncross_val=ncross_val)
        self.WiSARD_classify(ncross_val=ncross_val)

#}}}

class DataTools_reg:
    """
    - df : pandas.DataFrame
        schema : ['Relative Compactness', 'Surface Area', 'Wall Area',
                  'Roof Area', 'Overall Height', 'Orientation',
                  'Glazing Area', 'Glazing Area Distribution',
                  'Total Load']

    - df_bin : pandas.DataFrame, df but in list of binaries

    - X_train, X_test
    - y_train, y_test

    - X_bin_train, X_bin_test
    - y_bin_train, y_bin_test
    """

#{{{

    def __init__(self, path_DF=working_dir_path+'/energy_efficiency.csv'):
#{{{
        print '---------- Initiating DataTools_classif instance ----------'
        print 'Reading csv file ...'
        self.df = pd.read_csv(path_DF)

        print 'Generating df ...'
        # Drop Orientation variable which is useless for classification
        self.df = self.df.drop('Orientation', axis=1) # axis 0 for rows and 1 for columns

        # Sum of loads :
        self.df['Total Load'] = self.df['Heating Load'] + self.df['Cooling Load']
        self.df = self.df.drop(['Heating Load', 'Cooling Load'], axis=1)
        self.df = self.df.drop_duplicates()

        # Number of input variables :
        self.nX = self.df.shape[1] - 1

        print 'Generating df_bin ...'
        self.df_bin = self.to_reg_bin_df()

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

    def to_classif_bin_df(self):
#{{{
        df_bin = pd.DataFrame(data=None,
            columns=self.df.columns, index=self.df.index)

        n_reg_total = self.df.shape[0]

        for i in range(n_reg_total):
            for j in range(self.nX):
                float32 = self.df.iloc[i,j]
                float32_in_bin = float32_to_binary(float32)
                list_bin = bin_str_to_bin_list(float32_in_bin)
                df_bin.iloc[i,j] = list_bin

        df_bin['class'] = self.df['class']

        return df_bin
#}}}

    def to_reg_bin_df(self):
#{{{
        df_reg_bin = pd.DataFrame(data=None,
            columns=self.df.columns, index=self.df.index)

        n_reg_total = self.df.shape[0]

        # Input variables to binaries from DataFrame df :
        for i in range(n_reg_total):
            for j in range(self.nX):
                float32 = self.df.iloc[i,j]
                float32_in_bin = float32_to_binary(float32)
                list_bin = bin_str_to_bin_list(float32_in_bin)
                df_reg_bin.iloc[i,j] = list_bin

        # Outpit variable Total Load to list of bin :
        for i in range(n_reg_total):
            df_reg_bin.iloc[i,self.nX] = thermometer_float32_to_bin(
                self.df.iloc[i,self.nX])

        return df_reg_bin
#}}}


    def df_to_numpy_arrays(self, test_size=0.2):
#{{{
        n_reg_total = self.df.shape[0]

        X_train, X_test, y_train, y_test = \
            train_test_split(self.df.iloc[:,0:self.nX],
                             self.df.iloc[:,self.nX],
                             test_size=test_size)

        return X_train, X_test, y_train, y_test
#}}}

    def df_bin_to_numpy_arrays(self, test_size=0.2):
#{{{
        # flatten lists of bin :
        n_reg_total = self.df_bin.shape[0]
        X_bin = np.zeros( (n_reg_total, (self.nX-1)*32), dtype=int)

        for i in range(n_reg_total):
            list_tmp = []
            for j in range(self.nX-1):
                list_tmp = list_tmp + self.df_bin.iloc[i,j]
            X_bin[i,:] = np.array(list_tmp)

        # Constructing arrays randomly with train_test_split()
        X_bin_train, X_bin_test, y_bin_train, y_bin_test = \
            train_test_split(
            X_bin, self.df_bin.iloc[:, self.nX].as_matrix(),
            test_size=test_size)

        return X_bin_train, X_bin_test, y_bin_train, y_bin_test
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

    def WiSARD_classify(self, num_bits_addr=37, bleaching=False, **kwargs):
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

#}}}

    # - df_results : pandas.DataFrame of results obtained by classification
    #             ['Classifier', 'Characteristics', 'accuracy'])
    # - df_wisard  : results obtain from wisard Classifier
    #             ['Bleaching', 'num_bits_addr', 'accuracy'])




#-----------------------------------------------------

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

def thermometer_float32_to_bin(f, fmin=0, fmax=100, num_bits=100):
    list_of_bins = [0]*num_bits

    n_1_bits = int( round( f/(fmax - fmin)*float(num_bits) ) )
    for i in range(n_1_bits):
        list_of_bins[i] = 1

    return list_of_bins

def thermometer_bin_to_float32(list_of_bins, fmin=0, fmax=100):
    num_bits = len(list_of_bins)
    n_1_bits = sum(list_of_bins)
    f = float(n_1_bits)/float(num_bits)*float(fmax-fmin)

    return f


def get_pictures(dt_classif, dt_reg):
    df = dt_reg.df
    df.columns = ['X1','X2','X3','X4','X5', 'X7', 'X8', 'y']
    from random import randint
    # r1 = randint(0, df.shape[0])
    # r2 = randint(0, df.shape[0])
    # r3 = randint(0, df.shape[0])
    r1, r2, r3 = (0, 100, 200)
    fig, axes, _ = render_mpl_table(df.iloc[[r1,r2,r3],:],show_rowLabels=False)
    plt.tight_layout()

    df = dt_classif.df
    df.columns = ['X1','X2','X3','X4','X5', 'X7', 'X8', 'y']
    fig, axes, _ = render_mpl_table(df.iloc[[r1,r2,r3],:],
                                 show_rowLabels=False,
                                 header_col_color=[corange_2])
    plt.tight_layout()

    df = dt_classif.df_bin
    df.columns = ['X1','X2','X3','X4','X5', 'X7', 'X8', 'y']
    fig, axes, mpl_table = render_mpl_table(df.iloc[[r1,r2,r3],
                                                    [0,dt_classif.nX]],
                                 show_rowLabels=False,
                                 col_width=14,
                                 header_col_color=[corange_3])

    import six
    for k, cell in six.iteritems(mpl_table._cells):
        if k[1] == 0 :
            cell.set_width(14)
        if k[1] == 1 :
            cell.set_width(4)
    plt.tight_layout()

    return fig, axes

def get_wisard_parameters_graph(dt_classif):
    x = range(10,61)

    # for i in x:
    #     dt_classif.WiSARD_classify(num_bits_addr=i)
    #     dt_classif.WiSARD_classify(num_bits_addr=i, bleaching=True)

    fig, axes = plt.subplots()

    wis_no_bleach = dt_classif.df_wisard[
        (dt_classif.df_wisard['Bleaching']==False)]

    wis_with_bleach = dt_classif.df_wisard[
        (dt_classif.df_wisard['Bleaching']==True)]

    axes.plot(
        wis_no_bleach['num_bits_addr'], wis_no_bleach['accuracy'], '--o',
        wis_with_bleach['num_bits_addr'], wis_with_bleach['accuracy'], '--o')

    axes.legend(['Without Bleaching', 'With Bleaching'])
    axes.set_xlabel('num_bits_addr')
    axes.set_ylabel('accuracy')
    plt.tight_layout()






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

    dfs_class = DataTools_classif()
    dfs_reg = DataTools_reg()

    # dfs_class.GaussianProcessClassifier()
    # dfs_class.SupportVectorMachine()
    # dfs_class.NearestNeighbors()
    # dfs_class.MultiLayerPerceptron()
    # dfs_class.WiSARD_classify()

    dfs_class.calling_them_all(ncross_val=40)

    # get_pictures(dfs_class, dfs_reg)
    # get_wisard_parameters_graph(dfs_class)

    # from IPython import embed; embed() # Enter Ipython


    plt.show() # interactive plot

if __name__ == '__main__':
    main()
