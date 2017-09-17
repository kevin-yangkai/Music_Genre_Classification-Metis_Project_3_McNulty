import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA

from sklearn import naive_bayes
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from itertools import cycle

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style('white')
sns.set_palette('dark')
sns.set_context('talk')


def plot_PCA_2D(data, target, target_names):
    colors = cycle(['r','g','b','c','m','y','orange','w','aqua','yellow'])
    target_ids = range(len(target_names))
    plt.figure()
    for i, c, label in zip(target_ids, colors, target_names):
        print(i, '', target_names)
        plt.scatter(data[target == target_names[i], 0], data[target == target_names[i], 1], c=c, label=label)
    plt.legend()
    plt.show()


def save_dataframe_as_pickle(frame_to_save, save_name):
    with open(save_name, 'wb') as f:
        pickle.dump(frame_to_save, f)


def open_dataframe_pickle(name_of_pickle):
    with open(name_of_pickle, 'rb') as f:
        df_from_pickle = pickle.load(f)
    return df_from_pickle


def train_score(classifier, Xtrain, Xtest, ytrain, ytest):
    train_acc = classifier.score(Xtrain, ytrain)
    test_acc = classifier.score(Xtest, ytest)
    print("Training Data Accuracy: %0.2f" % (train_acc))
    print("Test Data Accuracy:     %0.2f" % (test_acc))
    ypred = classifier.predict(Xtest)
    conf = confusion_matrix(ytest, ypred)
    precision = (conf[0, 0] / (conf[0, 0] + conf[1, 0]))
    recall = (conf[0, 0] / (conf[0, 0] + conf[0, 1]))
    f1_score = 2 * ((precision * recall)/(precision + recall))
    print("Precision:              %0.2f" % precision)
    print("Recall:                 %0.2f" % recall)
    print("F1 Score:                 %0.2f" % f1_score)
    print('\n')


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Reds, labels=None):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    # plt.set_xticklabels(labels)
    # plt.yticks(labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



# import data
savefile_master_ZCR_final = 'tracks_set_2_ZCR_final.pickle'
savefile_master_RMS_final = 'tracks_set_2_RMS_final.pickle'
savefile_master_SPEC_CENTR_final = 'tracks_set_2_SPEC_CENTR_final.pickle'
savefile_master_SPEC_ROLL_final = 'tracks_set_2_SPEC_ROLL_final.pickle'
savefile_master_SPEC_FLUX_final = 'tracks_set_2_SPEC_FLUX_final.pickle'
savefile_master_MFCC_final = 'tracks_set_2_MFCC_final.pickle'
savefile_master_RYTHM_final = 'tracks_set_2_RYTHM_final.pickle'

# create the various empty dataframes and dump data into them from file.
df = pd.DataFrame()
df_info = pd.DataFrame()
df_ZCR = pd.DataFrame()
df_RMS = pd.DataFrame()
df_SPEC_CENTR = pd.DataFrame()
df_SPEC_ROLL = pd.DataFrame()
df_SPEC_FLUX = pd.DataFrame()
df_MFCC = pd.DataFrame()
df_RYTHM = pd.DataFrame()


data_temp = open_dataframe_pickle(savefile_master_ZCR_final)

df_info['labels'] = data_temp['label']
df['labels'] = data_temp['label']
df_info['tracks'] = data_temp['path']
df_info['sample_rate'] = data_temp['sample_rate']

df_ZCR['ZCR'] = data_temp['data']

data_temp = open_dataframe_pickle(savefile_master_RMS_final)
df_RMS['RMS'] = data_temp['data']

data_temp = open_dataframe_pickle(savefile_master_SPEC_CENTR_final)
df_SPEC_CENTR['SPEC_CENTR'] = data_temp['data']

data_temp = open_dataframe_pickle(savefile_master_SPEC_ROLL_final)
df_SPEC_ROLL['SPEC_ROLL'] = data_temp['data']

data_temp = open_dataframe_pickle(savefile_master_SPEC_FLUX_final)
df_SPEC_FLUX['SPEC_FLUX'] = data_temp['data']

data_temp = open_dataframe_pickle(savefile_master_MFCC_final)
df_MFCC['MFCCs_mean_1'] = data_temp['MFCCs_mean_1']
df_MFCC['MFCCs_std_1'] = data_temp['MFCCs_std_1']
df_MFCC['mspec_mean_1'] = data_temp['mspec_mean_1']
df_MFCC['mspec_std_1'] = data_temp['mspec_std_1']
df_MFCC['MFCCs_mean_2'] = data_temp['MFCCs_mean_2']
df_MFCC['MFCCs_std_2'] = data_temp['MFCCs_std_2']
df_MFCC['mspec_mean_2'] = data_temp['mspec_mean_2']
df_MFCC['mspec_std_2'] = data_temp['mspec_std_2']
df_MFCC['spec_mean_2'] = data_temp['spec_mean_2']
df_MFCC['spec_std_2'] = data_temp['spec_std_2']


data_temp = open_dataframe_pickle(savefile_master_RYTHM_final)
df_RYTHM['SSD'] = data_temp['SSD']
df_RYTHM['RP'] = data_temp['RP']
df_RYTHM['RH'] = data_temp['RH']

del data_temp

mean_temp = []
std_temp = []
for current in df_ZCR['ZCR']:
    mean_temp.append(np.mean(current))
    std_temp.append(np.std(current))

df['ZCR_mean'] = mean_temp
df['ZCR_std'] = std_temp

mean_temp = []
std_temp = []
for current in df_RMS['RMS']:
    mean_temp.append(np.mean(current))
    std_temp.append(np.std(current))

df['RMS_mean'] = mean_temp
df['RMS_std'] = std_temp

mean_temp = []
std_temp = []
for current in df_SPEC_CENTR['SPEC_CENTR']:
    mean_temp.append(np.mean(current))
    std_temp.append(np.std(current))

df['SPEC_CENTR_mean'] = mean_temp
df['SPEC_CENTR_std'] = std_temp

mean_temp = []
std_temp = []
for current in df_SPEC_ROLL['SPEC_ROLL']:
    mean_temp.append(np.mean(current))
    std_temp.append(np.std(current))

df['SPEC_ROLL_mean'] = mean_temp
df['SPEC_ROLL_std'] = std_temp

mean_temp = []
std_temp = []
for current in df_SPEC_FLUX['SPEC_FLUX']:
    mean_temp.append(np.mean(current))
    std_temp.append(np.std(current))

df['SPEC_FLUX_mean'] = mean_temp
df['SPEC_FLUX_std'] = std_temp


index = np.linspace(0, df.shape[0]-1, df.shape[0], dtype=np.int)
columns = (['MFCC_mean_01_1', 'MFCC_mean_02_1', 'MFCC_mean_03_1', 'MFCC_mean_04_1', 'MFCC_mean_05_1', 'MFCC_mean_06_1',
            'MFCC_mean_07_1', 'MFCC_mean_08_1', 'MFCC_mean_09_1', 'MFCC_mean_10_1', 'MFCC_mean_11_1', 'MFCC_mean_12_1',
            'MFCC_mean_13_1'])
df_temp = pd.DataFrame(columns=columns, index=index)
for i, current in enumerate(df_MFCC['MFCCs_mean_1']):
    # temp_list = pd.Series(current)
    temp_list = dict(zip(columns, pd.Series(current)))
    df_temp.iloc[i] = temp_list

df = pd.concat([df, df_temp], axis=1)


columns = (['MFCC_std_01_1', 'MFCC_std_02_1', 'MFCC_std_03_1', 'MFCC_std_04_1', 'MFCC_std_05_1', 'MFCC_std_06_1',
            'MFCC_std_07_1', 'MFCC_std_08_1', 'MFCC_std_09_1', 'MFCC_std_10_1', 'MFCC_std_11_1', 'MFCC_std_12_1',
            'MFCC_std_13_1'])
df_temp = pd.DataFrame(columns=columns, index=index)
for i, current in enumerate(df_MFCC['MFCCs_std_1']):
    # temp_list = pd.Series(current)
    temp_list = dict(zip(columns, pd.Series(current)))
    df_temp.iloc[i] = temp_list

df = pd.concat([df, df_temp], axis=1)

columns = (['mspec_mean_01_1', 'mspec_mean_02_1', 'mspec_mean_03_1', 'mspec_mean_04_1', 'mspec_mean_05_1', 'mspec_mean_06_1',
            'mspec_mean_07_1', 'mspec_mean_08_1', 'mspec_mean_09_1', 'mspec_mean_10_1', 'mspec_mean_11_1', 'mspec_mean_12_1',
            'mspec_mean_13_1'])
df_temp = pd.DataFrame(columns=columns, index=index)
for i, current in enumerate(df_MFCC['mspec_mean_1']):
    # temp_list = pd.Series(current)
    temp_list = dict(zip(columns, pd.Series(current)))
    df_temp.iloc[i] = temp_list

df = pd.concat([df, df_temp], axis=1)

columns = (['mspec_std_01_1', 'mspec_std_02_1', 'mspec_std_03_1', 'mspec_std_04_1', 'mspec_std_05_1', 'mspec_std_06_1',
            'mspec_std_07_1', 'mspec_std_08_1', 'mspec_std_09_1', 'mspec_std_10_1', 'mspec_std_11_1', 'mspec_std_12_1',
            'mspec_std_13_1'])
df_temp = pd.DataFrame(columns=columns, index=index)
for i, current in enumerate(df_MFCC['mspec_std_1']):
    # temp_list = pd.Series(current)
    temp_list = dict(zip(columns, pd.Series(current)))
    df_temp.iloc[i] = temp_list

df = pd.concat([df, df_temp], axis=1)

columns = (['MFCCs_mean_01_2', 'MFCCs_mean_02_2', 'MFCCs_mean_03_2', 'MFCCs_mean_04_2', 'MFCCs_mean_05_2', 'MFCCs_mean_06_2',
            'MFCCs_mean_07_2', 'MFCCs_mean_08_2', 'MFCCs_mean_09_2', 'MFCCs_mean_10_2', 'MFCCs_mean_11_2', 'MFCCs_mean_12_2',
            'MFCCs_mean_13_2'])
df_temp = pd.DataFrame(columns=columns, index=index)
for i, current in enumerate(df_MFCC['MFCCs_mean_2']):
    # temp_list = pd.Series(current)
    temp_list = dict(zip(columns, pd.Series(current)))
    df_temp.iloc[i] = temp_list

df = pd.concat([df, df_temp], axis=1)

columns = (['MFCCs_std_01_2', 'MFCCs_std_02_2', 'MFCCs_std_03_2', 'MFCCs_std_04_2', 'MFCCs_std_05_2', 'MFCCs_std_06_2',
            'MFCCs_std_07_2', 'MFCCs_std_08_2', 'MFCCs_std_09_2', 'MFCCs_std_10_2', 'MFCCs_std_11_2', 'MFCCs_std_12_2',
            'MFCCs_std_13_2'])
df_temp = pd.DataFrame(columns=columns, index=index)
for i, current in enumerate(df_MFCC['MFCCs_std_2']):
    # temp_list = pd.Series(current)
    temp_list = dict(zip(columns, pd.Series(current)))
    df_temp.iloc[i] = temp_list

df = pd.concat([df, df_temp], axis=1)

columns = (['mspec_mean_01_2', 'mspec_mean_02_2', 'mspec_mean_03_2', 'mspec_mean_04_2', 'mspec_mean_05_2', 'mspec_mean_06_2',
            'mspec_mean_07_2', 'mspec_mean_08_2', 'mspec_mean_09_2', 'mspec_mean_10_2', 'mspec_mean_11_2', 'mspec_mean_12_2',
            'mspec_mean_13_2'])
df_temp = pd.DataFrame(columns=columns, index=index)
for i, current in enumerate(df_MFCC['mspec_mean_2']):
    # temp_list = pd.Series(current)
    temp_list = dict(zip(columns, pd.Series(current)))
    df_temp.iloc[i] = temp_list

df = pd.concat([df, df_temp], axis=1)

columns = (['mspec_std_01_2', 'mspec_std_02_2', 'mspec_std_03_2', 'mspec_std_04_2', 'mspec_std_05_2', 'mspec_std_06_2',
            'mspec_std_07_2', 'mspec_std_08_2', 'mspec_std_09_2', 'mspec_std_10_2', 'mspec_std_11_2', 'mspec_std_12_2',
            'mspec_std_13_2'])
df_temp = pd.DataFrame(columns=columns, index=index)
for i, current in enumerate(df_MFCC['mspec_std_2']):
    # temp_list = pd.Series(current)
    temp_list = dict(zip(columns, pd.Series(current)))
    df_temp.iloc[i] = temp_list

df = pd.concat([df, df_temp], axis=1)

columns = (['spec_mean_01_2', 'spec_mean_02_2', 'spec_mean_03_2', 'spec_mean_04_2', 'spec_mean_05_2', 'spec_mean_06_2',
            'spec_mean_07_2', 'spec_mean_08_2', 'spec_mean_09_2', 'spec_mean_10_2', 'spec_mean_11_2', 'spec_mean_12_2',
            'spec_mean_13_2'])
df_temp = pd.DataFrame(columns=columns, index=index)
for i, current in enumerate(df_MFCC['spec_mean_2']):
    # temp_list = pd.Series(current)
    temp_list = dict(zip(columns, pd.Series(current)))
    df_temp.iloc[i] = temp_list

df = pd.concat([df, df_temp], axis=1)

columns = (['spec_std_01_2', 'spec_std_02_2', 'spec_std_03_2', 'spec_std_04_2', 'spec_std_05_2', 'spec_std_06_2',
            'spec_std_07_2', 'spec_std_08_2', 'spec_std_09_2', 'spec_std_10_2', 'spec_std_11_2', 'spec_std_12_2',
            'spec_std_13_2'])
df_temp = pd.DataFrame(columns=columns, index=index)
for i, current in enumerate(df_MFCC['spec_std_2']):
    # temp_list = pd.Series(current)
    temp_list = dict(zip(columns, pd.Series(current)))
    df_temp.iloc[i] = temp_list

df = pd.concat([df, df_temp], axis=1)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

label_vector = df['labels']
df = df.drop('labels', 1)


temp_insert=0

sns.set()
