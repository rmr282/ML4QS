from Chapter5.Clustering import NonHierarchicalClustering
from Chapter5.Clustering import HierarchicalClustering
from util.VisualizeDataset import VisualizeDataset

import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import argparse

DATA_PATH = Path('../intermediate_datafiles/')
DATASET_FNAME = 'chapter4_result.csv'
RESULT_FNAME = 'chapter5_result.csv'

try:
    dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
    dataset.index = pd.to_datetime(dataset.index)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

cmap = cm.get_cmap('tab10')
plt.figure(figsize=(6, 4))
plt.xlabel('k')
plt.ylabel('silhouette score')
plt.ylim((0, 1))

clusteringNH = NonHierarchicalClustering()
clusteringH = HierarchicalClustering()

k_values = range(2, 10)

# Do some initial runs to determine the right number for k

print('===== kmeans clustering =====')
silhouette_values = []
for k in k_values:
    print(f'k = {k}')
    dataset_cluster = clusteringNH.k_means_over_instances(copy.deepcopy(
        dataset), ['acc_phone_x', 'acc_phone_y', 'acc_phone_z'], k, 'default', 20, 10)
    silhouette_score = dataset_cluster['silhouette'].mean()
    print(f'silhouette = {silhouette_score}')
    silhouette_values.append(silhouette_score)

plt.plot(k_values, silhouette_values, color=cmap.colors[0], label='acc. kmeans')

silhouette_values = []
for k in k_values:
    print(f'k = {k}')
    dataset_cluster = clusteringNH.k_means_over_instances(copy.deepcopy(
        dataset), ['gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z'], k, 'default', 20, 10)
    silhouette_score = dataset_cluster['silhouette'].mean()
    print(f'silhouette = {silhouette_score}')
    silhouette_values.append(silhouette_score)

plt.plot(k_values, silhouette_values, color=cmap.colors[0], label='gyro kmeans', linestyle='dashed')

print('===== k medoids clustering =====')
silhouette_values = []
for k in k_values:
    print(f'k = {k}')
    dataset_cluster = clusteringNH.k_medoids_over_instances(copy.deepcopy(
        dataset), ['acc_phone_x', 'acc_phone_y', 'acc_phone_z'], k, 'default', 20, n_inits=10)
    silhouette_score = dataset_cluster['silhouette'].mean()
    print(f'silhouette = {silhouette_score}')
    silhouette_values.append(silhouette_score)

plt.plot(k_values, silhouette_values, color=cmap.colors[1], label='acc. k medoids')

print('===== k medoids clustering =====')
silhouette_values = []
for k in k_values:
    print(f'k = {k}')
    dataset_cluster = clusteringNH.k_medoids_over_instances(copy.deepcopy(
        dataset), ['gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z'], k, 'default', 20, n_inits=10)
    silhouette_score = dataset_cluster['silhouette'].mean()
    print(f'silhouette = {silhouette_score}')
    silhouette_values.append(silhouette_score)

plt.plot(k_values, silhouette_values, color=cmap.colors[1], label='gyro k medoids', linestyle='dashed')

print('===== agglomerative clustering =====')
silhouette_values = []
for k in k_values:
    print(f'k = {k}')
    dataset, l = clusteringH.agglomerative_over_instances(dataset, [
        'acc_phone_x', 'acc_phone_y', 'acc_phone_z'], k, 'euclidean', use_prev_linkage=True, link_function='ward')
    silhouette_score = dataset_cluster['silhouette'].mean()
    print(f'silhouette = {silhouette_score}')
    silhouette_values.append(silhouette_score)

plt.plot(k_values, silhouette_values, color=cmap.colors[2], label='acc. agglomerative')

print('===== agglomerative clustering =====')
silhouette_values = []
for k in k_values:
    print(f'k = {k}')
    dataset, l = clusteringH.agglomerative_over_instances(dataset, [
        'gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z'], k, 'euclidean', use_prev_linkage=True, link_function='ward')
    silhouette_score = dataset_cluster['silhouette'].mean()
    print(f'silhouette = {silhouette_score}')
    silhouette_values.append(silhouette_score)

plt.plot(k_values, silhouette_values, color=cmap.colors[2], label='gyro agglomerative', linestyle='dashed')
plt.legend()

plt.show()
