import copy
import numpy as np
import pandas as pd
from ast import literal_eval

from Chapter5.Clustering import NonHierarchicalClustering, HierarchicalClustering
from util.VisualizeDataset import VisualizeDataset

FILENAME = 'sensorlog_readouts_transposed_d250(2)_freq.csv'

data = pd.read_csv(FILENAME)
data.drop('label', inplace=True, axis=1)
data.drop('label_id', inplace=True, axis=1)

DataViz = VisualizeDataset(__file__ + '_sensorlog')

clusteringNH = NonHierarchicalClustering()
clusteringH = HierarchicalClustering()

basic_features = ['gravity_x', 'gravity_y', 'gravity_z',
                  'lsm6dsm_accelerometer_x', 'lsm6dsm_accelerometer_y', 'lsm6dsm_accelerometer_z',
                  'lsm6dsm_gyroscope_x', 'lsm6dsm_gyroscope_y', 'lsm6dsm_gyroscope_z',
                  'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
                  'ak09915_magnetometer_x', 'ak09915_magnetometer_y', 'ak09915_magnetometer_z']

cluster_features = ['linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z']

k_values = [4]
silhouette_values = []

for k in k_values:
    print(f'k = {k}')
    dataset_cluster = clusteringNH.k_medoids_over_instances(
        copy.deepcopy(data), cluster_features, k, 'default', 20, n_inits=10)
    silhouette_score = dataset_cluster['silhouette'].mean()
    print(f'silhouette = {silhouette_score}')
    silhouette_values.append(silhouette_score)

DataViz.plot_xy(x=[k_values], y=[silhouette_values], xlabel='k', ylabel='silhouette score',
                ylim=[0, 1], line_styles=['b-'])

k = k_values[np.argmax(silhouette_values)]

k = 4
dataset_kmed = clusteringNH.k_medoids_over_instances(
    copy.deepcopy(data), cluster_features, k, 'default', 20, n_inits=50)
DataViz.plot_clusters_3d(dataset_kmed, cluster_features, 'cluster', ['label'])
