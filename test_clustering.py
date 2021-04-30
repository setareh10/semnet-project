#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:05:45 2020

@author: sr05
"""
from sklearn.cluster import KMeans
import os
import mne
import time
import pickle
import numpy as np
import sn_config as C
from joblib import Parallel, delayed
from SN_semantic_ROIs import SN_semantic_ROIs
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from sklearn.model_selection import (cross_validate, KFold)
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import explained_variance_score as exvar
from sklearn import preprocessing
# from keras.models import Sequential
# from keras.layers import Dense
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects = C.subjects
MRI_sub = C.subjects_mri
# Parameters
snr = C.snr
lambda2 = C.lambda2_epoch
label_path = C.label_path
SN_ROI = SN_semantic_ROIs()
# ROI_x=1
# ROI_y=0
s = time.time()
fs = 1000
f_down_sampling = 20  # 100Hz, 20Hz
t_down_sampling = fs/f_down_sampling  # 10ms, 50ms
i = 10

ROI_x = 2
ROI_y = 3
cond = 'fruit'
normalize = False
meg = subjects[i]
sub_to = MRI_sub[i][1:15]

# morph labels from fsaverage to each subject
labels = mne.morph_labels(SN_ROI, subject_to=sub_to,
                          subject_from='fsaverage', subjects_dir=data_path)

# read epochs
epo_name = data_path + meg + 'block_'+cond+'_words_epochs-epo.fif'

epochs_cond = mne.read_epochs(epo_name, preload=True)

# crop epochs
epochs = epochs_cond['words'].copy(
).crop(-.200, .900).resample(f_down_sampling)

inv_fname_epoch = data_path + meg + 'InvOp_'+cond+'_EMEG-inv.fif'


output = [0]*2
# read inverse operator,apply inverse operator
inv_op = read_inverse_operator(inv_fname_epoch)
stc = apply_inverse_epochs(epochs, inv_op, lambda2, method='MNE',
                           pick_ori="normal", return_generator=False)

for j, idx in enumerate([ROI_x, ROI_y]):
    labels[idx].subject = sub_to
    # define dimentions of matrix (vertices X timepoints), & initializing
    v, t = stc[0].in_label(labels[idx]).data.shape
    X = np.zeros([v, len(stc), t])
    # create output array of size (vertices X stimuli X timepoints)
    for s in np.arange(0, len(stc)):
        S = stc[s].in_label(labels[idx]).data
        X[:, s, :] = S

    output[j] = X
x = output[0]
y = output[1]
X = np.zeros([x.shape[0], x.shape[1], x.shape[0]])
Y = np.zeros([y.shape[0], y.shape[1], y.shape[0]])

scaler = MinMaxScaler(feature_range=(-1, 1))
for t in np.arange(x.shape[-1]):
    X[:, :, t] = scaler.fit(x[:, :, t]).transform(x[:, :, t])
    Y[:, :, t] = scaler.fit(y[:, :, t]).transform(y[:, :, t])

model = KMeans()
visualizer = KElbowVisualizer(model, k=(2, 20))

visualizer.fit(X[:, :, 10])
visualizer.elbow_value_       # Fit the data to the visualizer
# visualizer.show()        # Finalize and render the figure


# define the model
model = KMeans(n_clusters=visualizer.elbow_value_)
# fit the model
model.fit(X[:, :, 10])
# assign a cluster to each example
yhat = model.predict(X[:, :, 10])
# retrieve unique clusters
clusters = np.unique(yhat)
# create scatter plot for samples from each cluster
X_clusters = []
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = np.where(yhat == cluster)
    # create scatter of these samples
    X_clusters.append(row_ix)


data = X[X_clusters[5], :, 10].reshape(9, X.shape[1])
pca = PCA(n_components=1)
# pca.fit(data.transpose())
pc1 = pca.fit_transform(data.transpose()).transpose()
corr = []
for k in range(9):
    corr.append(np.corrcoef(pc1, data[k, :])[0, 1])

u = np.corrcoef(data)
cluster_mean = data.mean(0)
cluster_var = data.var(1)
np.where(cluster_var == cluster_var.max())
