import math
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval

from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TemporalAbstraction import NumericalAbstraction

labels = pd.read_csv('../datasets/MS4QS_sensor_labels_day1.csv', sep='|')

u_labels = np.array([labels['label'].values[i] for i in sorted(np.unique(labels['label'], return_index=True)[1])])
labels = {k: v for (k, v) in zip(*labels.to_dict('list').values())}

data = pd.read_csv('../datasets/MS4QS_sensor_data_day1.csv', sep='|')
data['statusId'] = data['statusId'].map(labels)
data.sort_values('timestamp', inplace=True)

data['timestamp'] = data['timestamp'].astype('datetime64[ms]')
data.set_index('timestamp', drop=False, inplace=True)

sensors = data.groupby('sensorName')
sensor_groups = {
    'single_val_sensors': [],  # ['APDS-9940 Light', 'HSPPAD042A Pressure'],
    'xyz_sensors': ['Linear Acceleration', 'Gravity', 'LSM6DSM Gyroscope',
                    'LSM6DSM Accelerometer', 'AK09915 Magnetometer']
}


def make_proxy(zvalue, scalar_mappable, **kwargs):
    color = scalar_mappable.cmap(scalar_mappable.norm(zvalue))
    return Line2D([0, 1], [0, 1], color=color, **kwargs)


delta = 250
delta_td = pd.Timedelta(milliseconds=delta)

# --------------
# Assignment 4.3 Temporal
# --------------
lin_acc = sensors.get_group('Linear Acceleration').copy()
lin_acc['value'] = lin_acc['value'].apply(lambda x: literal_eval(x))
lin_acc = lin_acc.join(pd.DataFrame(lin_acc['value'].to_list(), columns=['x', 'y', 'z'], index=lin_acc.index))

mask = np.ones(len(lin_acc), dtype=bool)
last_ts = lin_acc.iloc[0].timestamp
for i, (_, ts) in enumerate(lin_acc.timestamp[1:].iteritems()):
    if ts - last_ts < delta_td:
        mask[i] = False
    else:
        last_ts = ts

lin_acc = lin_acc[mask]

df = lin_acc.x

cmap = cm.get_cmap('tab10')
norm = BoundaryNorm(range(len(np.unique(lin_acc['statusId'])) + 1), cmap.N)

NumAbs = NumericalAbstraction()
milliseconds_per_instance = (lin_acc.iloc[1].timestamp - lin_acc.iloc[0].timestamp).microseconds / 1000

window_sizes = [int(float(5000) / milliseconds_per_instance),
                int(float(0.5 * 60000) / milliseconds_per_instance),
                int(float(5 * 60000) / milliseconds_per_instance)]

fig, axs = plt.subplots(6, figsize=(12, 12), sharex='all')

inxval = mdates.date2num(lin_acc.index.to_pydatetime())
x_ax = np.linspace(inxval[0], inxval[-1], len(df))
points = np.array([x_ax, df.values]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(np.array([np.where(u_labels == x)[0][0] for x in lin_acc['statusId']]))

axs[0].add_collection(lc)
axs[0].set_xlim((lin_acc[lin_acc.x.notnull()].index.min(), lin_acc[lin_acc.x.notnull()].index.max()))
axs[0].set_ylim((math.floor(df.min() * 1.1), math.ceil(df.max() * 1.1)))
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

legend = [make_proxy(item, lc, linewidth=2) for item in list(range(len(u_labels)))]
axs[0].legend(legend, u_labels, loc='upper center', bbox_to_anchor=(.5, -.05), ncol=6, fancybox=True, shadow=True)

for ws in window_sizes:
    mean = df.rolling(ws).mean()
    axs[1].plot(x_ax, mean.values, label=f"mean (window size {ws})")
axs[1].set_ylim((-1, 3))
axs[1].set_title('mean', loc='left')
axs[1].legend(loc='upper center', bbox_to_anchor=(.5, -.05), ncol=3, fancybox=True, shadow=True)

for ws in window_sizes:
    std = df.rolling(ws).std()
    axs[2].plot(x_ax, std.values, label=f"st. dev. (window size {ws})")
axs[2].set_ylim((0, 15))
axs[2].set_title('st.dev.', loc='left')
axs[2].legend(loc='upper center', bbox_to_anchor=(.5, -.05), ncol=3, fancybox=True, shadow=True)

# --------------
# Assignment 4.3 Frequency
# --------------
FreqAbs = FourierTransformation()

fs = float(1000) / milliseconds_per_instance
ws = int(float(10000) / milliseconds_per_instance)

freq_data = FreqAbs.abstract_frequency(lin_acc, ['x'], ws, fs)

print(f"Calculating correlations between features and label...")
freq_data['label'] = pd.factorize(freq_data.statusId)[0]
pearson_corr = freq_data.iloc[:, 4:].corr('pearson')['label'][:]
spearman_corr = freq_data.iloc[:, 4:].corr('spearman')['label'][:]
corrs = pd.concat([pearson_corr, spearman_corr], keys=['Pearson', 'Spearman'],
                  axis=1).sort_values(['Pearson', 'Spearman'], ascending=False)

print("Name\tPearson\tSpearman")
for corr in corrs.iterrows():
    print(f"{corr[0]}:\t{corr[1].Pearson}\t{corr[1].Spearman}")

axs[3].plot(x_ax, freq_data.x_max_freq)
axs[3].set_title('x max. frequency', loc='left')

axs[4].plot(x_ax, freq_data.x_freq_weighted)
axs[4].set_yscale('log')
axs[4].set_title('x frequency weighted (log)', loc='left')

axs[5].plot(x_ax, freq_data.x_pse)
axs[5].set_title('x pse', loc='left')

fig.suptitle(fr"Linear Acceleration x-axis ($\Delta${delta}ms)")
plt.show()

# --------------
# Assignment 2.1
# --------------
for val_type, sensor_names in sensor_groups.items():
    for sensor_name in sensor_names:
        group = sensors.get_group(sensor_name).copy()
        group['value'] = group['value'].apply(lambda x: literal_eval(x))

        group = group.join(pd.DataFrame(group['value'].to_list(),
                                        columns=['x', 'y', 'z'] if val_type == 'xyz_sensors' else ['val'],
                                        index=group.index))

        mask = np.ones(len(group), dtype=bool)
        last_ts = group.iloc[0].timestamp
        for i, (_, ts) in enumerate(group.timestamp[1:].iteritems()):
            if ts - last_ts < delta_td:
                mask[i] = False
            else:
                last_ts = ts

        group = group[mask]

        cmap = cm.get_cmap('tab10')
        norm = BoundaryNorm(range(len(np.unique(group['statusId'])) + 1), cmap.N)

        if val_type == 'xyz_sensors':
            fig, axs = plt.subplots(3, figsize=(12, 8), sharex='all')
            for i, df in enumerate([group.x, group.y, group.z]):
                inxval = mdates.date2num(group.index.to_pydatetime())
                points = np.array([np.linspace(inxval[0], inxval[-1], len(df)), df.values]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                lc = LineCollection(segments, cmap=cmap, norm=norm)
                lc.set_array(np.array([np.where(u_labels == x)[0][0] for x in group['statusId']]))

                axs[i].add_collection(lc)
                axs[i].set_xlim((group[(group.x.notnull() | group.y.notnull() | group.z.notnull())].index.min(),
                                 group[(group.x.notnull() | group.y.notnull() | group.z.notnull())].index.max()))
                axs[i].set_ylim((math.floor(df.min() * 1.1), math.ceil(df.max() * 1.1)))
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

                axs[i].set_title(df.name, loc='left')

                if i == 2:
                    legend = [make_proxy(item, lc, linewidth=2) for item in list(range(len(u_labels)))]
                    axs[i].legend(legend, u_labels, loc='upper center', bbox_to_anchor=(.5, -.25),
                                  ncol=6, fancybox=True, shadow=True)
        else:
            fig, ax = plt.subplots(figsize=(8, 4))

            points = np.array([group.index.values, group['val'].values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(np.array([np.where(u_labels == x)[0][0] for x in group['statusId']]))

            ax.add_collection(lc)
            ax.set_xlim((group[group.val.notnull()].index.min(), group[group.val.notnull()].index.max()))
            ax.set_ylim((math.floor(group.val.min() * 1.1), math.ceil(group.val.max() * 1.1)))

            legend = [make_proxy(item, lc, linewidth=2) for item in list(range(len(u_labels)))]
            ax.legend(legend, u_labels, loc='upper center', bbox_to_anchor=(.5, -.25),
                      ncol=6, fancybox=True, shadow=True)

        fig.suptitle(fr"{sensor_name} ($\Delta${delta}ms)")
        plt.show()
