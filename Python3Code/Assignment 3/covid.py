import numpy as np
import pandas as pd
import scipy.interpolate as interp
from matplotlib import pyplot as plt

hr_data = pd.read_csv('./data/heart_rate.csv')
hr_data.datetime = pd.to_datetime(hr_data.datetime)
hr_data = hr_data.sort_values('datetime').drop('is_resting', axis=1)

bp_data = pd.read_csv('./data/blood_pressure.csv')
bp_data.rename(columns={'measurement_datetime': 'datetime'}, inplace=True)
bp_data.datetime = pd.to_datetime(bp_data.datetime)
bp_data = bp_data.sort_values('datetime')

survey_data = pd.read_csv('./data/surveys.csv')
survey_data = survey_data[survey_data.scale == 'S_COVID_OVERALL'][['created_at', 'user_code', 'value']]
survey_data.rename(columns={'created_at': 'datetime', 'value': 'covid_symptoms_score'}, inplace=True)
survey_data.datetime = pd.to_datetime(survey_data.datetime)
survey_data = survey_data.sort_values('datetime')

hrv_data = pd.read_csv('./data/hrv_measurements.csv')
hrv_data.rename(columns={'measurement_datetime': 'datetime'}, inplace=True)
hrv_data.datetime = pd.to_datetime(hrv_data.datetime)
hrv_data = hrv_data.sort_values('datetime')
hrv_data = hrv_data[['user_code', 'datetime', 'meanrr', 'mxdmn', 'sdnn', 'rmssd', 'pnn50',
                     'mode', 'amo', 'lf', 'hf', 'vlf', 'lfhf', 'total_power']]

combi_data = pd.merge_asof(hr_data, survey_data, on='datetime', by='user_code')
combi_data = combi_data.dropna(subset=['covid_symptoms_score'])

combi_data = pd.merge_asof(combi_data, bp_data, on='datetime', by='user_code', direction='nearest')
combi_data = pd.merge_asof(combi_data, hrv_data, on='datetime', by='user_code')

combi_data.set_index('datetime', inplace=True)


for user_code, user_data in combi_data.groupby('user_code'):
    if len(user_data) > 99 and len(np.unique(user_data.covid_symptoms_score)) > 1:
        user_data = user_data.copy()

        plt.plot(user_data.index, user_data.heart_rate, label='heart rate')
        plt.plot(user_data.index, user_data.diastolic, label='bp diastolic')
        plt.plot(user_data.index, user_data.systolic, label='bp systolic')
        plt.plot(user_data.index, user_data.covid_symptoms_score, label='covid_symptoms_score')

        plt.show()

        # Remove duplicated blood pressure data and interpolate to smooth
        bp_cols = ['diastolic', 'systolic', 'functional_changes_index', 'circulatory_efficiency',
                   'kerdo_vegetation_index', 'robinson_index']
        user_data.loc[user_data.diastolic.diff(-1).fillna(user_data.diastolic) == 0, bp_cols] = np.nan

        for col in bp_cols:
            epoch = pd.Timestamp('1970-01-01')
            bp_data = user_data[col].dropna()
            bp_data.index = (bp_data.index - epoch) // pd.Timedelta('1s')

            t, c, k = interp.splrep(bp_data.index, bp_data.values, s=0, k=4)
            xx = np.linspace(bp_data.index.min(), bp_data.index.max(), 100)
            spline = interp.BSpline(t, c, k, extrapolate=False)

            # plt.plot(x, y, 'bo', label='Original points')
            plt.plot(xx, spline(xx), 'r', label='BSpline')
            plt.grid()
            plt.legend(loc='best')
            plt.show()

            user_data[col] = user_data[col].interpolate()
            user_data[col] = user_data[col].fillna(method='bfill')

        plt.plot(user_data.index, user_data.diastolic, label='bp diastolic interp.')
        plt.plot(user_data.index, user_data.systolic, label='bp systolic interp.')

        # Outlier detection


        # Increase interval size and cute impute (i.e. make up data)
        delta = 30  # seconds
        time_groups = user_data.groupby(pd.Grouper(freq=f"{delta}s"))

        # Lowpass filtering


        # Add frequency features


        # Clustering


        # Saving the world from covid using sick multi-user ML

        plt.title(user_code)
        plt.legend()
        plt.show()

        print(f"\n\nCorrelations for user {user_code} ({len(user_data)})")
        pearson_corr = user_data.corr('pearson')['covid_symptoms_score'][:]
        spearman_corr = user_data.corr('spearman')['covid_symptoms_score'][:]
        corrs = pd.concat([pearson_corr, spearman_corr], keys=['Pearson', 'Spearman'], axis=1).sort_values(
            ['Pearson', 'Spearman'], ascending=False)

        print("Name,Pearson,Spearman")
        for corr in corrs.iterrows():
            print(f"{corr[0]},{corr[1].Pearson},{corr[1].Spearman}")
