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

import pandas as pd
import seaborn as sns

import numpy as np

from PIL import Image # Python Image Library
from StringIO import StringIO # manage IO strings & path

# Classifications :
from sklearn.metrics import accuracy_score, roc_auc_score,\
                            precision_score, recall_score, \
                            confusion_matrix, classification_report

from sklearn.gaussian_process import GaussianProcessClassifier

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier


# Global variables :
script_path = os.path.abspath(sys.argv[0])
default_csv_dir = os.path.dirname(script_path)+"/mnist/"

default_blocks = 4 # for color space diminution

# For ipython init :
limit_train_set = 30
limit_test_set = 10

class wrapperDataFrame:
    """
    - df_train : pandas.DataFrame of train set of numbers
        schema : ['jpg_path', 'number', 'im_data']
    - X_train, y_train

    - df_test : pandas.DataFrame of test set of numbers
        schema : ['jpg_path', 'number', 'im_data']
    - X_test, y_test

    - df_results : pandas.DataFrame of results obtained by classification
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

        # print "Constructing df_train columns of pixel values ..."
        # self.fill_pixel_columns(self.df_train)

        # print "Constructing df_test columns of pixel values ..."
        # self.fill_pixel_columns(self.df_test)

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

        # Initiate df_results :
        self.df_results = pd.DataFrame(columns=['Classifier', 'accuracy'])
#}}}

    def fill_pixel_columns(self, df):
#{{{
        ''' Given a DataFrames with ['jpg_path'] column, it adds columns
            'pix0', 'pix1', 'pix2', ... 'pixN' and fills them pixel by pixel,
            storing line by line of pixels.

            It does pretty  much the same as :
            self.df_train['im_data'] = [np.array(Image.open(path).getdata())
                    for path in self.df_train['jpg_path']]
            but results in columns of values instead of 1 column of np.arrays.

            Args:
                df (pandas.DataFrame): df to be modified

            Remarks:
                It is wayyyy much slower than the other solution proposed
                in the description.
        '''

        df_irow = 0
        for path in df['jpg_path']:
            image = Image.open(path)
            nx, ny = image.size

            # Stored by lines : k = j + i*ny
            for i in range(nx):
                for j in range(ny):
                    k = j + i*ny
                    col_str = "pix" + str(k)
                    df.loc[df_irow, col_str] = \
                        image.getpixel(xy=(i,j))

            df_irow += 1
#}}}


    def GaussianProcessClassifier(self):
#{{{
        print "GaussianProcessClassifier :"
        gp = GaussianProcessClassifier()

        print "Training ..."
        gp.fit(self.X_train, self.y_train)

        print "Testing ..."
        y_predicted = gp.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_predicted)
        print "accuracy = ", accuracy
        self.df_results.loc[self.df_results.shape[0]] = \
            ['GaussianProcess', accuracy]
#}}}


# SPECTRES :
#{{{
def get_spectre(image, blocks=default_blocks):
#{{{
    '''Given a PIL Image object it returns its feature vector.

    Args:
      image (PIL.Image): image to process.
      blocks (int, optional): number of block to subdivide the “L”
        (luminance : for greyscale images) space into.

    Returns:
      list of float: feature vector if successful.
    '''

    if image.mode == 'L':
        blocks = blocks ** 3 # precision mutliplied by 3
        feature = [0] * blocks
        pixel_count = 0
        for pixel in image.getdata():
            idx = int(pixel/(256/blocks))
            feature[idx] += 1
            pixel_count += 1
        return[float(x)/pixel_count for x in feature]

    elif image.mode == 'RGB':
        feature = [0] * blocks * blocks * blocks
        pixel_count = 0
        for pixel in image.getdata():
            ridx = int(pixel[0]/(256/blocks))
            gidx = int(pixel[1]/(256/blocks))
            bidx = int(pixel[2]/(256/blocks))
            idx = ridx + gidx * blocks + bidx * blocks * blocks
            feature[idx] += 1
            pixel_count += 1
        return [float(x)/pixel_count for x in feature]
    else:
        return(None)
#}}}

def get_spectre_file(image_path):
#{{{
    '''Given an image path it returns its feature vector.

    Args:
      image_path (str): path of the image file to process.

    Returns:
      list of float: feature vector on success, None otherwise.
    '''
    image_fp = StringIO(open(image_path, 'rb').read())
    try:
        image = Image.open(image_fp)
        return get_spectre(image)
    except IOError:
        return None
#}}}
#}}}

def display_numbers(df, blocks=default_blocks):
#{{{
    '''Given an dataframe it return fig, axes of PIL.Images in Luminance
    (greyscale L) and bilevel spaces.

    Args:
      df (pandas.DataFrame): pandas dataframe.

    Returns:
      list of float: feature vector on success, None otherwise.
    '''
    nrows = df.shape[0]
    fig, axes = plt.subplots(2, nrows, figsize=(15,6), facecolor='w',
            edgecolor='k')
    fig.subplots_adjust(hspace = .2, wspace=.2)
    fig.suptitle("Number and Luminance Spectre", fontsize=14)
    axes = axes.ravel()

    for i in range(nrows):
        axes[i].imshow(Image.open(df['jpg_path'][i]))
        axes[i].set_title("Num of rowid " + str(i))
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)

    for i in range(nrows):
        # x = np.arange(0, len(df['arr_lum'][i]), 1)
        # axes[i+nrows].set_title("Lum Spectre of rowid " + str(i))
        # axes[i+nrows].plot(x, df['arr_lum'][i])
        axes[i+nrows].imshow(Image.open(df['jpg_path'][i]).convert('1'))
        axes[i+nrows].get_xaxis().set_visible(False)
        axes[i+nrows].get_yaxis().set_visible(False)

    return fig, axes
#}}}


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

    # DataFrames Initialization :
    dfs = wrapperDataFrame(limit_train_set, limit_test_set)
    print dfs.df_train.head()

    # Classifications :
    print
    dfs.GaussianProcessClassifier()

    # Results :
    print
    print dfs.df_results



if __name__ == "__main__":
    sys.exit(main())
