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

from PIL import Image, ImageOps # Python Image Library
from StringIO import StringIO # manage IO strings & path

# Classifications :
from sklearn.metrics import accuracy_score, roc_auc_score,\
                            precision_score, recall_score, \
                            confusion_matrix, classification_report

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier

from PyWANN.WiSARD import WiSARD

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier


##############################################
# Global variables :
script_path = os.path.abspath(sys.argv[0])
working_dir_path = os.path.dirname(script_path)

default_csv_dir = working_dir_path+"/mnist/"

#Cmap
from matplotlib.colors import ListedColormap
my_cmap = ListedColormap(sns.color_palette("OrRd", 10).as_hex())
my_cmap_conf_mat = ListedColormap(sns.color_palette("OrRd", 20).as_hex())
my_cmap_2 = ListedColormap(sns.color_palette("RdYlBu", 10).as_hex())

#Some colors:
color_green = sns.color_palette('GnBu', 10)[3]
color_blue = sns.color_palette("PuBu", 10)[7]
color_purple = sns.color_palette("PuBu", 10)[2]
color_red = sns.color_palette("OrRd", 10)[6]

default_cfg_fig = {'bool': False, 'path': working_dir_path+"/figures/" \
                  ,'figsize': (10,10), 'fout_name': 'out'}
##############################################


class wrapperDataFrame:
    """
    - df_train : pandas.DataFrame of train set of numbers
        schema : ['jpg_path', 'number', 'im_data']

    - df_test : pandas.DataFrame of test set of numbers
        schema : ['jpg_path', 'number', 'im_data']

    - X_train, y_train
    - X_test, y_test
    - X_train_bin, X_test_bin
    - threshold : default None
                  if X_train_bin and X_test_bin already exists, threshold=int

    - df_results : pandas.DataFrame of results obtained by classification
                ['Classifier', 'Threshold', 'Characteristics', 'accuracy'])
    - df_wisard  : results obtain from wisard Classifier
                ['Bleaching', 'Threshold', 'num_bits_addr', 'accuracy'])
    """

    def __init__(self, limit_train_set, limit_test_set, csv_dir=default_csv_dir):
#{{{
        print "Reading ", csv_dir+ "train-labels.csv ..."
        self.df_train = pd.read_csv(csv_dir+"train-labels.csv",
                names=["jpg_path", "number"], nrows=limit_train_set)

        print "Reading ", csv_dir+ "test-labels.csv ..."
        self.df_test = pd.read_csv(csv_dir+"test-labels.csv",
                names=["jpg_path", "number"], nrows=limit_test_set)

        print "Adjusting jpg_path ..."
        self.df_train['jpg_path'] = csv_dir + self.df_train['jpg_path']
        self.df_test['jpg_path'] = csv_dir + self.df_test['jpg_path']


        print "Constructing im_data ..."
        self.df_train['im_data'] = [np.array(Image.open(path).getdata())
                for path in self.df_train['jpg_path']]
        self.df_test['im_data'] = [np.array(Image.open(path).getdata())
                for path in self.df_test['jpg_path']]

        print "Copying im_data to numpy arrays X_train, y_train ..."
        self.X_train = np.array([row for row in self.df_train['im_data']])
        self.y_train = self.df_train['number'].as_matrix()

        print "Copying im_data to numpy arrays X_test, y_test ..."
        self.X_test = np.array([row for row in self.df_test['im_data']])
        self.y_test = self.df_test['number'].as_matrix()

        # Default values :
        self.threshold = None

        # Initiate df_results :
        self.df_results = pd.DataFrame(columns=
                ['Classifier', 'Threshold', 'Characteristics', 'accuracy'])

        self.df_wisard = pd.DataFrame(columns=
                ['Bleaching', 'Threshold', 'num_bits_addr', 'accuracy'])
#}}}


    def binarize(self, threshold=256/2):
        if threshold != self.threshold:
            binarize = lambda t: 0 if t<threshold else 1
            self.X_train_bin = np.vectorize(binarize)(self.X_train)
            self.X_test_bin = np.vectorize(binarize)(self.X_test)
            self.threshold = threshold

    def results_to_barplot(self, cfg_fig=default_cfg_fig):
        fig, ax = plt.subplots(figsize=cfg_fig['figsize'])
        ax.set_ylabel('Scores')



    ##############################################
    # Classifiers :

    def GaussianProcessClassifier(self, threshold=None):
#{{{
        print "GaussianProcessClassifier :"
        gp = GaussianProcessClassifier()

        if threshold is None:
            X_train = self.X_train
            X_test = self.X_test
        else:
            if threshold != self.threshold:
                self.binarize(threshold)
            X_train = self.X_train_bin
            X_test = self.X_test_bin

        print "Training ..."
        gp.fit(X_train, self.y_train)

        print "Testing ..."
        y_predicted = gp.predict(X_test)

        accuracy = accuracy_score(self.y_test, y_predicted)
        self.df_results.loc[self.df_results.shape[0]] = \
            ['GaussianProcess', threshold, '', accuracy]

        return y_predicted
#}}}

    def SupportVectorMachine(self, threshold=None):
#{{{
        print "Support Vector Machine :"
        clf = svm.SVC()

        if threshold is None:
            X_train = self.X_train
            X_test = self.X_test
        else:
            if threshold != self.threshold:
                self.binarize(threshold)
            X_train = self.X_train_bin
            X_test = self.X_test_bin

        print "Training ..."
        clf.fit(X_train, self.y_train)

        print "Testing ..."
        y_predicted = clf.predict(X_test)

        accuracy = accuracy_score(self.y_test, y_predicted)
        self.df_results.loc[self.df_results.shape[0]] = \
            ['Support_Vector_Machine', threshold, '', accuracy]

        return y_predicted
#}}}

    def NearestNeighbors(self, threshold=None):
#{{{
        print "NearestNeighborsClassifier :"
        clf = neighbors.KNeighborsClassifier(n_neighbors=10)

        if threshold is None:
            X_train = self.X_train
            X_test = self.X_test
        else:
            if threshold != self.threshold:
                self.binarize(threshold)
            X_train = self.X_train_bin
            X_test = self.X_test_bin

        print "Training ..."
        clf.fit(X_train, self.y_train)

        print "Testing ..."
        y_predicted =clf.predict(X_test)

        accuracy = accuracy_score(self.y_test, y_predicted)
        self.df_results.loc[self.df_results.shape[0]] = \
            ['NearestNeighbors', threshold, 'n_neighbors=10', accuracy]

        return y_predicted
#}}}

    def MultiLayerPerceptron(self, hidden_layer_sizes=(100), threshold=None):
#{{{
        print "MultiLayerPerceptronClassifier :"
        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)

        charac = 'hidden_layer_sizes : ' + str(hidden_layer_sizes)

        if threshold is None:
            X_train = self.X_train
            X_test = self.X_test
        else:
            if threshold != self.threshold:
                self.binarize(threshold)
            X_train = self.X_train_bin
            X_test = self.X_test_bin

        print "Training ..."
        clf.fit(X_train, self.y_train)

        print "Testing ..."
        y_predicted =clf.predict(X_test)

        accuracy = accuracy_score(self.y_test, y_predicted)
        self.df_results.loc[self.df_results.shape[0]] = \
            ['MultiLayerPerceptron', threshold, charac, accuracy]

        return y_predicted
#}}}

    def WiSARD(self, threshold=45, num_bits_addr=27, **kwargs):

        self.binarize(threshold)
        print "WiSARD : threshold="+str(threshold)+' ; num_bits_addr='+str(num_bits_addr)
        clf = WiSARD(num_bits_addr=num_bits_addr, **kwargs)

        charac = '\n num_bits_addr : ' + str(num_bits_addr)

        print "Training ..."
        clf.fit(self.X_train_bin, self.y_train)

        print "Testing..."
        y_predicted = clf.predict(self.X_test_bin)

        accuracy = accuracy_score(self.y_test, y_predicted)
        # self.df_results.loc[self.df_results.shape[0]] = \
        #     ['WiSARD_without_bleaching', threshold, charac, accuracy]
        self.df_wisard.loc[self.df_wisard.shape[0]] = \
            [kwargs['bleaching'], threshold, num_bits_addr, accuracy]

        return y_predicted


##############################################
# Pre Processing analysis functions :

# class wrapperSubplots:
#     """
#     - cfg_fig : dictionary
#         { 'bool': if True will save figures,
#           'path': path to output directory,
#           'figsize' : size of the output figure,
#           'fout_name': filename,
#         }

#     """

#     def __init__(self, cfg_fig=default_cfg_fig):
#     fig, axes = plt.subplots(2, nrows, figsize=(15,6), facecolor='w',
#             edgecolor='k')

def display_figures_luminance(df, cfg_fig=default_cfg_fig):
#{{{
    nrows = df.shape[0]
    fig, axes = plt.subplots(1, nrows, figsize=cfg_fig['figsize'],
            facecolor='w', edgecolor='k')

    # fig.subplots_adjust(hspace = .2, wspace=.2) # spaces between subplots

    # Convert array_like into contiguous flattened array to be able to loop :
    axes = axes.ravel()

    for i in range(nrows):
        axes[i].imshow(Image.open(df['jpg_path'][i]))
        axes[i].set_title(str(i)+".jpg")
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)

    if cfg_fig['bool']:
        filename = cfg_fig['path'] + cfg_fig['fout_name']
        plt.savefig(filename, bbox_inches='tight')

    return fig, axes
#}}}

def display_figures_with_tresh(path, thresh1=45, thresh2=200,
        cfg_fig=default_cfg_fig):
#{{{
    fig, axes = plt.subplots(1, 3, figsize=(15,6), facecolor='w',
            edgecolor='k')
    fig.subplots_adjust(hspace = .2, wspace=.2)
    axes = axes.ravel()

    axes[0].imshow(Image.open(path))
    axes[0].set_title(os.path.basename(path) + "\n Luminance space"
            , fontsize=14)
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)

    axes[1].imshow(ImageOps.invert(Image.open(path))
            .point(lambda x:0 if x<thresh1 else 255, '1'))
    axes[1].set_title(os.path.basename(path) +
            "\n Binarized with threshold = " + str(thresh1), fontsize=14)
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)

    axes[2].imshow(ImageOps.invert(Image.open(path))
            .point(lambda x:0 if x<thresh2 else 255, '1'))
    axes[2].set_title(os.path.basename(path) +
            "\n Binarized with threshold = " + str(thresh2), fontsize=14)
    axes[2].get_xaxis().set_visible(False)
    axes[2].get_yaxis().set_visible(False)

    if cfg_fig['bool']:
        filename = cfg_fig['path'] + cfg_fig['fout_name']
        plt.savefig(filename, bbox_inches='tight')

    return fig, axes
#}}}

def array_to_figaxis(arr, axis):
#{{{
    ''' Fill the axis given with the image represented by the array.

    Returns: ax
    '''
    nxny = arr.shape[0]
    nx = np.uint8(math.sqrt(nxny))
    arr_reshaped = arr.reshape(nx,nx)

    # Open as PIL image
    image = Image.fromarray(np.uint8(arr_reshaped))

    axis.imshow(image)
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

    return axis
#}}}

def mean_by_pixel_image(X, cfg_fig=default_cfg_fig):
#{{{
    ''' Compute a mean of each pixel of X_train and plot it.

    Returns:
    fig, ax
    '''

    # Init
    n_jpg = X.shape[0]
    n_pix = X.shape[1]
    arr_mean = np.zeros(n_pix, dtype=np.uint8)

    # Compute mean vector
    for j in range(n_pix):
        arr_mean[j] = np.uint8((X[:,j].mean()))

    fig, ax = plt.subplots(figsize=(5,5))

    ax = array_to_figaxis(arr_mean, ax)

    # Making a frame :
    ax.grid(False)
    size = fig.get_size_inches()
    ax.axvline(0, color='k')
    ax.axvline(27, color='k')
    ax.axhline(0, color='k')
    ax.axhline(27, color='k')

    if cfg_fig['bool']:
        filename = cfg_fig['path'] + cfg_fig['fout_name']
        plt.savefig(filename, bbox_inches='tight')

    return fig, ax
#}}}

def plot_dist_mat(X, y, cfg_fig=default_cfg_fig):
#{{{
    ''' Plot the distance matrix using Euclidiean norm.

    Returns:
    fig, ax
    '''

    # Sorting according to clusters to make then apparent :
    M = np.concatenate( (X, y[:, np.newaxis]),
                        axis=1)
    # Sort according to last column :
    M = M[M[:,-1].argsort()]
    M = M[0:-1] # remove last column

    from scipy.spatial.distance import pdist, squareform
    dist_mat = pdist(M, 'euclidean')
    dist_mat = squareform(dist_mat) #translates this flattened form into a full matrix

    fig, ax = plt.subplots(figsize=cfg_fig['figsize'])

    im = ax.imshow(dist_mat, cmap=my_cmap, interpolation='none')

    # get colorbar smaller than matrix
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # want a more natural, table-like display
    ax.invert_yaxis()

    # Move top xaxes :
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.axis('off')

    if cfg_fig['bool']:
        filename = cfg_fig['path'] + cfg_fig['fout_name']
        plt.savefig(filename, bbox_inches='tight')


    return fig, ax
#}}}

def display_barplot_of_classes(y_train, y_test, cfg_fig=default_cfg_fig):
#{{{
    ''' Barplot of y_train and y_test values frequencies.

    Returns:
    fig, ax
    '''
    fig, ax = plt.subplots(figsize=cfg_fig['figsize'])

    ind = np.arange(0,10,1) # the x locations for the groups
    width = 0.35       # the width of the bars

    class_count_train = np.zeros(10)
    for i in y_train:
        class_count_train[i] += 1

    ind = np.arange(0,10,1)
    bars_train = ax.bar(ind, class_count_train, width, color=color_blue)

    class_count_test = np.zeros(10)
    for i in y_test:
        class_count_test[i] += 1

    bars_test = ax.bar(ind+width, class_count_test, width, color=color_red)

    ax.set_xlabel('Digits')
    ax.set_ylabel('Number of digits by classe')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(ind)
    ax.legend((bars_train[0], bars_test[0]),
            ('Train : '+str(y_train.shape[0])+' registers',
            'Test : '+str(y_test.shape[0])+' registers'))

    if cfg_fig['bool']:
        filename = cfg_fig['path'] + cfg_fig['fout_name']
        plt.savefig(filename, bbox_inches='tight')

    return fig, ax
#}}}




##############################################
# POST Processing analysis functions :

def plot_confusion_mat(y_true, y_predicted, cmap=plt.cm.Blues,
                       cfg_fig=default_cfg_fig):
#{{{
    ''' Plot the confusion matrix.

    Returns:
    fig, ax
    '''
    M = confusion_matrix(y_true, y_predicted)

    fig, ax = plt.subplots(figsize=cfg_fig['figsize'])

    classes = np.arange(10)

    # Realizing offset on xticks and yticks but keeping horizontal and
    # verticale whites lines by removing grid objets then add lines objets
    xticks_arr = np.arange(0,M.shape[0],1)
    yticks_arr =  np.arange(0,M.shape[1],1)

    # plt.xticks(xticks_arr, xticks_arr)
    # plt.yticks(yticks_arr, yticks_arr)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)#, rotation=45)
    plt.yticks(tick_marks, classes)

    ax.grid(False)
    for j in xticks_arr:
        ax.axvline(j+0.5, color='w')
    for i in yticks_arr:
        ax.axhline(i+0.5, color='w')

    im = ax.imshow(M, cmap=cmap, interpolation='nearest')

    plt.colorbar(im, fraction=0.046, pad=0.04) # get colorbar smaller than matrix

    # plt.colorbar()

    # want a more natural, table-like display
    ax.invert_yaxis()

    # Move top xaxes :
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.xlabel('Predicted Value')
    plt.ylabel('True Value')

    # Adding the number in each cell :
    thresh = M.max() / 2.
    for i, j in itertools.product(range(M.shape[0]), range(M.shape[1])):
        plt.text(j, i, M[i, j],
                 horizontalalignment="center",
                 color="white" if M[i, j] > thresh else "black")

    if cfg_fig['bool']:
        filename = cfg_fig['path'] + cfg_fig['fout_name']
        plt.savefig(filename, bbox_inches='tight')

    return fig, ax
#}}}

def display_surface3D_wisard(df_wisard, cfg_fig=default_cfg_fig):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = df_wisard['Threshold'].values
    Y = df_wisard['num_bits_addr'].values
    Z = df_wisard['accuracy'].values

    ax.set_xlabel('Threshold')
    ax.set_ylabel('num_bits_addr')
    ax.set_zlabel('accuracy')

    ax.set_zlim(0., 1.)

    for x, y, z in [(X, Y, Z)]:
        ax.scatter(x, y, z, c=-z, cmap=my_cmap_2, marker='o')

    # fig.colorbar(ax, shrink=0.5, aspect=5)

    return fig, ax

def setup_argparser():
#{{{
    """ Define and return the command argument parser. """
    parser = argparse.ArgumentParser(description='''Redes Neurais Sem Peso.''')

    parser.add_argument('--csv_dir', dest='csv_dir', required=False,
                        default=default_csv_dir, action='store',
                        help='''Path to directory that contains csv files.
                        ''')

    parser.add_argument('--limit_train_set', dest='limit_train_set',
                        required=False,
                        action='store', default=None, type=int,
                        help='''Number maximum of registers in csv train
                        set to consider''')

    parser.add_argument('--limit_test_set', dest='limit_test_set',
                        required=False,
                        action='store', default=None, type=int,
                        help='''Number maximum of registers in csv test
                        set to consider''')

    parser.add_argument('--pre_processing', dest='pre_processing',
                        required=False,
                        action='store_true', default=False,
                        help='''Whether to run pre_processing or not''')

    parser.add_argument('-v', '--verbose', dest='verbose', required=False,
                        default=0, type=int,
                        help='Verbose level: 0 for errors, 1 for info or 2 for debug.')

    return parser
#}}}


def main(argv=None):

    parser = setup_argparser()

    try:
        args = parser.parse_args()

    except argparse.ArgumentError as exc:
        log.exception('Error parsing options.')
        parser.error(str(exc.message))
        raise

    verbose  = args.verbose
    csv_dir = args.csv_dir
    limit_train_set = args.limit_train_set
    limit_test_set = args.limit_test_set
    pre_processing = args.pre_processing

    # Verifying existing directories :
    cfg_fig = default_cfg_fig
    cfg_fig['bool'] = True
    if cfg_fig['bool']:
        try:
            os.makedirs(cfg_fig['path'])
        except OSError as exc: # Python >2.5
            if exc.errno != errno.EEXIST:
                raise

    # DataFrames Initialization :
    dfs = wrapperDataFrame(limit_train_set, limit_test_set)
    # print dfs.df_train.head()

    # Pre processing analysis :
    if pre_processing:
        print "----------------------------------------"
        print "Pre Processing : "

        print "Plotting first figures in Luminance ..."
        cfg_fig['fout_name'] = "first_8_jpgs_luminance"
        cfg_fig['figsize'] = (14,7)
        fig, axes = display_figures_luminance(dfs.df_train[:8], cfg_fig)

        print "Plotting figures with threshold ..."
        cfg_fig['fout_name'] = "0_jpg_with_threshold"
        cfg_fig['figsize'] = (14,7)
        fig, axes = display_figures_with_tresh(dfs.df_train['jpg_path'][0]
                , cfg_fig=cfg_fig)

        print "Plotting Mean of Pixel Picture ..."
        cfg_fig['fout_name'] = "mean_pix_n" + str(limit_train_set)\
            + str(limit_train_set) + " jpgs"
        fig, ax = mean_by_pixel_image(dfs.X_train, cfg_fig)

        print "Plotting Distance Matrix ..."
        cfg_fig['fout_name'] = "dist_mat_n" + str(limit_train_set)\
            + str(limit_train_set)+" jpgs"
        fig, ax = plot_dist_mat(dfs.X_train, dfs.y_train, cfg_fig)

        print "Plotting barplot of classes ..."
        cfg_fig['fout_name'] = "barplot_of_classes_ntrain"\
            +str(limit_train_set)+"_ntest"+str(limit_test_set)
        fig, ax = display_barplot_of_classes(dfs.y_train, dfs.y_test,
                cfg_fig=cfg_fig)


    # Classifications :
    print "----------------------------------------"
    print "Classifications : "
    # Some results without binarizing the pictures :
    dfs.GaussianProcessClassifier()
    dfs.SupportVectorMachine()
    dfs.NearestNeighbors()
    dfs.MultiLayerPerceptron()
    fig, ax = display_barplot_of_classes(dfs.y_train, dfs.y_test,
            cfg_fig=cfg_fig)

    # Some results with binarization of the pictures :
    dfs.NearestNeighbors(threshold=15)
    dfs.MultiLayerPerceptron(threshold=50)
    y_predicted = dfs.WiSARD(threshold=45, num_bits_addr=27, bleaching=True)


    # Printing results results :
    print "----------------------------------------"
    print dfs.df_results
    print "----------------------------------------"
    print dfs.df_wisard

    print "Plotting Confusion Matrix for WiSARD with bleaching and \
threshold=45, num_bits_addr=27 ..."
    cfg_fig['fout_name'] = "conf_mat_wisard_bleach_t45_n27"
    fig, ax = plot_confusion_mat(dfs.y_test, y_predicted, cfg_fig=cfg_fig)


    from IPython import embed; embed() # Enter Ipython

    # # For parameters calibration purpose :
    # for t in range(10,150,5):
    #     y_predicted = dfs.NearestNeighbors(threshold=t)

    # for t in range(10,150,5):
    #     for h in range(50, 150, 5):
    #         y_predicted = dfs.MultiLayerPerceptron(
    #                 threshold=t, hidden_layer_sizes=(h))

    # for t in range(10,150,5):
    #     for n in range(1,64):
    #         y_predicted = dfs.WiSARD(threshold=t, num_bits_addr=n,
    #                 bleaching=False)

    # for t in range(10,150,5):
    #     for n in range(1,64):
    #         y_predicted = dfs.WiSARD(threshold=t, num_bits_addr=n,
    #                 bleaching=True)



    # Obtaining surface3D pictures :
    # dfs.df_wisard = pd.read_csv('wisard_results_without_bleaching.csv')
    # # fig, ax = display_surface3D_wisard(dfs.df_wisard.iloc[10:500:5])
    # fig, ax = display_surface3D_wisard(dfs.df_wisard)

    # dfs.df_wisard = pd.read_csv('wisard_results_with_bleaching.csv')
    # # fig, ax = display_surface3D_wisard(dfs.df_wisard.iloc[10:500:5])
    # fig, ax = display_surface3D_wisard(dfs.df_wisard)
    # from IPython import embed; embed() # Enter Ipython




if __name__ == "__main__":
    sys.exit(main())
