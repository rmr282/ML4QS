import copy
import numpy as np
import os
import pandas as pd
from ast import literal_eval
from tqdm import tqdm

from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection, DistanceBasedOutlierDetection
from Chapter3.DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from Chapter4.TemporalAbstraction import NumericalAbstraction, CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter5.Clustering import NonHierarchicalClustering
from util.VisualizeDataset import VisualizeDataset


def unnesting(df, explode, axis):
    if axis == 1:
        df1 = pd.concat([df[x].explode() for x in explode], axis=1)
        return df1.join(df.drop(explode, 1), how='left')
    else:
        dfs = [pd.DataFrame(df[x].tolist(), columns=['_x', '_y', '_z'] if len(df[x].to_list()[0]) > 1 else ['_val'],
                            index=df.index).add_prefix(x) for x in explode]
        if dfs:
            df1 = pd.concat(dfs, axis=1)
            return df1.join(df.drop(explode, 1), how='left')
        else:
            return None

FILENAME = 'sensorlog_readouts_transposed_d250(2)'
RECREATE = False

DataViz = VisualizeDataset(__file__ + '_sensorlog')

OutlierDistr = DistributionBasedOutlierDetection()
OutlierDist = DistanceBasedOutlierDetection()
MisVal = ImputationMissingValues()
LowPass = LowPassFilter()
PCA = PrincipalComponentAnalysis()
NumAbs = NumericalAbstraction()
FreqAbs = FourierTransformation()
CatAbs = CategoricalAbstraction()
clusteringNH = NonHierarchicalClustering()

labels = pd.read_csv('../datasets/MS4QS_sensor_labels_day1.csv', sep='|')

u_labels = np.array([labels['label'].values[i] for i in sorted(np.unique(labels['label'], return_index=True)[1])])
labels = {k: v for (k, v) in zip(*labels.to_dict('list').values())}

delta = 250
delta_td = pd.Timedelta(milliseconds=delta)

if RECREATE or not os.path.exists(FILENAME + '_raw.csv'):
    data = pd.read_csv('../datasets/MS4QS_sensor_data_day1.csv', sep='|')
    data['label'] = data['statusId'].map(labels)
    data['label_id'] = pd.factorize(data.label)[0]

    data.sort_values('timestamp', inplace=True)

    data['timestamp'] = data['timestamp'].astype('datetime64[ms]')
    data.set_index(pd.DatetimeIndex(data['timestamp']), drop=False, inplace=True)

    time_groups = data.groupby(pd.Grouper(freq=f"{delta}ms"))

    nones = 0
    sensor_readouts = None
    for name, group in tqdm(time_groups, desc="*Beep Boop* Filtering data based on time interval..."):
        if not group.empty:
            d_readouts = group.groupby('sensorName').first().reset_index()
            d_readouts['value'] = d_readouts['value'].apply(lambda x: literal_eval(x))

            d_vals = {k.lower().replace(' ', '_'): d_readouts[d_readouts['sensorName'] == k].value.iloc[0]
                      for k in d_readouts.sensorName}
            d_vals = unnesting(pd.Series(d_vals).to_frame().T, d_vals.keys(), axis=0)

            d_vals['label'] = d_readouts.label.iloc[0]
            d_vals['timestamp'] = d_readouts.timestamp.iloc[0]
            d_vals = d_vals.set_index('timestamp', drop=True)

            sensor_readouts = d_vals if sensor_readouts is None else sensor_readouts.append(d_vals)
        else:
            nones += 1

    print(f"Done. Number of None rows ignored: {nones}")
    data = sensor_readouts
    data = data.join(pd.get_dummies(data.label, prefix='label'))

    data.to_csv(FILENAME + '_raw.csv')
else:
    data = pd.read_csv(FILENAME + '_raw.csv')
    if 'Unnamed: 0' in data.columns:
        data.drop('Unnamed: 0', inplace=True, axis=1)

selected_predictor_cols = [c for c in data.columns if not 'label' in c and c != 'timestamp']
periodic_measurements = ['gravity_x', 'gravity_y', 'gravity_z',
                         'lsm6dsm_accelerometer_x', 'lsm6dsm_accelerometer_y', 'lsm6dsm_accelerometer_z',
                         'lsm6dsm_gyroscope_x', 'lsm6dsm_gyroscope_y', 'lsm6dsm_gyroscope_z',
                         'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
                         'ak09915_magnetometer_x', 'ak09915_magnetometer_y', 'ak09915_magnetometer_z']

if RECREATE or not os.path.exists(FILENAME + '_outlier.csv'):
    print("*Reutel Reutel* Doing Chauvenet outlier detection...")
    for col in selected_predictor_cols:
        print(f'Measurement is now: {col}')
        data = OutlierDistr.chauvenet(data, col)
        data.loc[data[f'{col}_outlier'] == True, col] = np.nan
        del data[col + '_outlier']

    data.to_csv(FILENAME + '_outlier.csv')
else:
    data = pd.read_csv(FILENAME + '_outlier.csv')
    if 'Unnamed: 0' in data.columns:
        data.drop('Unnamed: 0', inplace=True, axis=1)

if RECREATE or not os.path.exists(FILENAME + '_lowpass.csv'):
    print("*Pruttel Pruttel* Doing Low Pass filtering")
    for col in selected_predictor_cols:
        data = MisVal.impute_interpolate(data, col)

    fs = float(1000) / delta
    cutoff = 1.5
    for col in periodic_measurements:
        data = LowPass.low_pass_filter(data, col, fs, cutoff, order=10)
        data[col] = data[col + '_lowpass']
        del data[col + '_lowpass']

    for col in selected_predictor_cols:
        data = MisVal.impute_interpolate(data, col)

    n_pcs = np.argmax(PCA.determine_pc_explained_variance(data, selected_predictor_cols)) + 1
    data = PCA.apply_pca(copy.deepcopy(data), selected_predictor_cols, n_pcs)

    data.to_csv(FILENAME + '_lowpass.csv')
else:
    data = pd.read_csv(FILENAME + '_lowpass.csv')
    if 'Unnamed: 0' in data.columns:
        data.drop('Unnamed: 0', inplace=True, axis=1)

if RECREATE or not os.path.exists(FILENAME + '_freq.csv'):
    print("*Prrrt Prrrt* Adding frequency features...")
    if 'timestamp' in data.columns:
        data = data.set_index('timestamp', drop=True)
    data.index = pd.to_datetime(data.index)

    ws = int(float(0.5 * 60000) / delta)
    fs = float(1000) / delta

    for col in selected_predictor_cols:
        aggregations = data[col].rolling(f"{ws}s", min_periods=ws)
        data[col + '_temp_mean_ws_' + str(ws)] = aggregations.mean()
        data[col + '_temp_std_ws_' + str(ws)] = aggregations.std()

    data['label_id'] = pd.factorize(data.label)[0]
    data = CatAbs.abstract_categorical(data, ['label_id'], ['like'], 0.03, int(float(5 * 60000) / delta), 2)
    data = FreqAbs.abstract_frequency(copy.deepcopy(data), periodic_measurements,
                                      int(float(10000) / 250), float(1000) / 250)

    # window_overlap = 0.9
    # skip_points = int((1-window_overlap) * ws)
    # data = data.iloc[::skip_points, :]

    data.to_csv(FILENAME + '_freq.csv')
else:
    data = pd.read_csv(FILENAME + '_freq.csv')
    if 'Unnamed: 0' in data.columns:
        data.drop('Unnamed: 0', inplace=True, axis=1)

if RECREATE or not os.path.exists(FILENAME + '_cluster.csv'):
    print("*Homm Humm* Adding clustering features...")
    clusteringNH = NonHierarchicalClustering()

    k = 4
    data = clusteringNH.k_means_over_instances(data, ['linear_acceleration_x', 'linear_acceleration_y',
                                                      'linear_acceleration_z'], k, 'default', 50, 50)
    del data['silhouette']

    data.to_csv(FILENAME + '_cluster.csv')
else:
    data = pd.read_csv(FILENAME + '_cluster.csv')
    if 'Unnamed: 0' in data.columns:
        data.drop('Unnamed: 0', inplace=True, axis=1)
