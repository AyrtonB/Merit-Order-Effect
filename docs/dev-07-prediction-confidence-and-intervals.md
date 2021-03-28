# Prediction & Confidence Intervals



[![Binder](https://notebooks.gesis.org/binder/badge_logo.svg)](https://notebooks.gesis.org/binder/v2/gh/AyrtonB/Merit-Order-Effect/main?filepath=nbs%2Fdev-07-prediction-confidence-and-intervals.ipynb)

This notebook outlines the calculation of the prediction and confidence intervals for the GB and DE price MOE models

<br>

### Imports

```python
import numpy as np
import pandas as pd

import pickle

import matplotlib.pyplot as plt

from moepy import lowess, eda
from moepy.surface import PicklableFunction

from ipypb import track
```

<br>

### Great Britain

We'll start by loading and cleaning the data for GB

```python
df_EI = eda.load_EI_df('../data/raw/electric_insights.csv')
df_EI_model = df_EI[['day_ahead_price', 'demand', 'solar', 'wind']].dropna()

s_price = df_EI_model['day_ahead_price']
s_dispatchable = df_EI_model['demand'] - df_EI_model[['solar', 'wind']].sum(axis=1)
```

<br>

We'll then calculate the estimate for the 68% prediction interval

```python
def get_pred_intvl(low_q_fp, high_q_fp):
    """Calculates the prediction interval between the low and high quantile models specified"""
    smooth_dates_low = pickle.load(open(low_q_fp, 'rb'))
    smooth_dates_high = pickle.load(open(high_q_fp, 'rb'))

    x_pred = np.linspace(3, 61, 581)
    dt_pred = pd.date_range('2009-01-01', '2020-12-31', freq='1D')

    df_pred_low = smooth_dates_low.predict(x_pred=x_pred, dt_pred=dt_pred)
    df_pred_low.index = np.round(df_pred_low.index, 1)

    df_pred_high = smooth_dates_high.predict(x_pred=x_pred, dt_pred=dt_pred)
    df_pred_high.index = np.round(df_pred_high.index, 1)

    df_pred_intvl = df_pred_high - df_pred_low
    
    return df_pred_intvl
```

```python
%%time

df_pred_68pct_intvl_GB = get_pred_intvl('../data/models/DAM_price_GB_p16.pkl', '../data/models/DAM_price_GB_p84.pkl')

df_pred_68pct_intvl_GB.head()
```

    Wall time: 11.4 s
    




|   Unnamed: 0 |   2009-01-01 |   2009-01-02 |   2009-01-03 |   2009-01-04 |   2009-01-05 |   2009-01-06 |   2009-01-07 |   2009-01-08 |   2009-01-09 |   2009-01-10 | ...   |   2020-12-22 |   2020-12-23 |   2020-12-24 |   2020-12-25 |   2020-12-26 |   2020-12-27 |   2020-12-28 |   2020-12-29 |   2020-12-30 |   2020-12-31 |
|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|:------|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
|          3   |     -4.77878 |     -4.80147 |     -4.82393 |     -4.84614 |     -4.86811 |     -4.88982 |     -4.91126 |     -4.93241 |     -4.95325 |     -4.97378 | ...   |      41.4778 |      41.4841 |      41.4904 |      41.4967 |      41.503  |      41.5093 |      41.5157 |      41.522  |      41.5284 |      41.5348 |
|          3.1 |     -4.73778 |     -4.76051 |     -4.78301 |     -4.80526 |     -4.82727 |     -4.84902 |     -4.8705  |     -4.89169 |     -4.91257 |     -4.93314 | ...   |      41.3044 |      41.3107 |      41.317  |      41.3233 |      41.3296 |      41.3359 |      41.3422 |      41.3486 |      41.3549 |      41.3613 |
|          3.2 |     -4.69656 |     -4.71933 |     -4.74186 |     -4.76415 |     -4.7862  |     -4.80799 |     -4.82951 |     -4.85074 |     -4.87167 |     -4.89228 | ...   |      41.1312 |      41.1375 |      41.1437 |      41.15   |      41.1563 |      41.1626 |      41.169  |      41.1753 |      41.1816 |      41.188  |
|          3.3 |     -4.65507 |     -4.67787 |     -4.70044 |     -4.72276 |     -4.74485 |     -4.76668 |     -4.78824 |     -4.80951 |     -4.83047 |     -4.85113 | ...   |      40.9582 |      40.9645 |      40.9707 |      40.977  |      40.9833 |      40.9896 |      40.9959 |      41.0023 |      41.0086 |      41.0149 |
|          3.4 |     -4.61326 |     -4.63609 |     -4.65869 |     -4.68105 |     -4.70317 |     -4.72504 |     -4.74664 |     -4.76794 |     -4.78895 |     -4.80964 | ...   |      40.7855 |      40.7918 |      40.798  |      40.8043 |      40.8106 |      40.8169 |      40.8232 |      40.8295 |      40.8358 |      40.8421 |</div>



<br>

We can see that we get some quantile crossing at the extreme ends of the dispatch curve which is why some of our 68% interval values are negative, to counter this we'll weight our prediction interval by how often that part of the dispatch curve is where the price clears at.

```python
s_pred_idx_weight = s_dispatchable.round(1).value_counts().sort_index()
dispatchable_gen_idxs = sorted(list(set(s_pred_idx_weight.index).intersection(df_pred_68pct_intvl_GB.index)))

pred_68pct_intvl = np.average(df_pred_68pct_intvl_GB.mean(axis=1).loc[dispatchable_gen_idxs], weights=s_pred_idx_weight.loc[dispatchable_gen_idxs])

print(f'The 68% prediction interval for GB is {round(pred_68pct_intvl, 2)} £/MWh')
```

    The 68% prediction interval for GB is 16.32 £/MWh
    

<br>

We'll use our bootstrapping helper function to calculate the confidence interval of the GB model

```python
%%capture

center_dts = pd.date_range(s_price.index.min(), s_price.index.max(), freq='3MS') + pd.Timedelta(days=45)

all_conf_intvl_95pct = []

for center_dt in track(center_dts):
    s_price_subset = s_price[center_dt-pd.Timedelta(days=45):center_dt+pd.Timedelta(days=45)]
    s_dispatchable_subset = s_dispatchable[center_dt-pd.Timedelta(days=45):center_dt+pd.Timedelta(days=45)]

    df_bootstrap = lowess.bootstrap_model(s_price_subset.values, s_dispatchable_subset.values, num_runs=100, frac=0.3, num_fits=10)
    conf_intvl_95pct = df_bootstrap.replace(0, np.nan).quantile([0.025, 0.975], axis=1).diff().dropna(how='all').mean(axis=1).iloc[0]
    
    all_conf_intvl_95pct += [conf_intvl_95pct]
    
conf_intvl_95pct_GB = np.array(all_conf_intvl_95pct).mean()
```

```python
print(f'The 95% confidence interval for GB is {round(conf_intvl_95pct_GB, 2)} £/MWh')
```

    The 95% confidence interval for GB is 1.03 £/MWh
    

<br>

### Germany

We'll start by loading and cleaning the data for DE

```python
%%time

df_DE = eda.load_DE_df('../data/raw/energy_charts.csv', '../data/raw/ENTSOE_DE_price.csv')

df_DE_model = df_DE[['price', 'demand', 'Solar', 'Wind']].dropna()

s_DE_price = df_DE_model['price']
s_DE_demand = df_DE_model['demand']
s_DE_dispatchable = df_DE_model['demand'] - df_DE_model[['Solar', 'Wind']].sum(axis=1)
```

    Wall time: 1.72 s
    

<br>

We'll then calculate the estimate for the 68% prediction interval

```python
%%time

df_pred_68pct_intvl_DE = get_pred_intvl('../data/models/DAM_price_DE_p16.pkl', '../data/models/DAM_price_DE_p84.pkl')

s_pred_idx_weight = s_DE_dispatchable.round(1).value_counts().sort_index()
dispatchable_gen_idxs = sorted(list(set(s_pred_idx_weight.index).intersection(df_pred_68pct_intvl_DE.index)))

pred_68pct_intvl = np.average(df_pred_68pct_intvl_DE.mean(axis=1).loc[dispatchable_gen_idxs], weights=s_pred_idx_weight.loc[dispatchable_gen_idxs])

print(f'The 68% prediction interval for DE is {round(pred_68pct_intvl, 2)} EUR/MWh')
```

    The 68% prediction interval for DE is 13.79 EUR/MWh
    Wall time: 1.5 s
    

<br>

We'll use our bootstrapping helper function to calculate the confidence interval of the GB model

```python
%%capture

center_dts = pd.date_range(s_DE_price.index.min(), s_DE_price.index.max(), freq='3MS') + pd.Timedelta(days=45)

all_conf_intvl_95pct = []

for center_dt in track(center_dts):
    s_price_subset = s_DE_price[center_dt-pd.Timedelta(days=45):center_dt+pd.Timedelta(days=45)]
    s_dispatchable_subset = s_DE_dispatchable[center_dt-pd.Timedelta(days=45):center_dt+pd.Timedelta(days=45)]

    df_bootstrap = lowess.bootstrap_model(s_price_subset.values, s_dispatchable_subset.values, num_runs=100, frac=0.3, num_fits=10)
    conf_intvl_95pct = df_bootstrap.replace(0, np.nan).quantile([0.025, 0.975], axis=1).diff().dropna(how='all').mean(axis=1).iloc[0]
    
    all_conf_intvl_95pct += [conf_intvl_95pct]
    
conf_intvl_95pct_DE = np.array(all_conf_intvl_95pct).mean()
```

```python
print(f'The 95% confidence interval for DE is {round(conf_intvl_95pct_DE, 2)} EUR/MWh')
```

    The 95% confidence interval for DE is 1.69 EUR/MWh
    
