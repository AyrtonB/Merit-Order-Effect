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
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2009-01-01</th>
      <th>2009-01-02</th>
      <th>2009-01-03</th>
      <th>2009-01-04</th>
      <th>2009-01-05</th>
      <th>2009-01-06</th>
      <th>2009-01-07</th>
      <th>2009-01-08</th>
      <th>2009-01-09</th>
      <th>2009-01-10</th>
      <th>...</th>
      <th>2020-12-22</th>
      <th>2020-12-23</th>
      <th>2020-12-24</th>
      <th>2020-12-25</th>
      <th>2020-12-26</th>
      <th>2020-12-27</th>
      <th>2020-12-28</th>
      <th>2020-12-29</th>
      <th>2020-12-30</th>
      <th>2020-12-31</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3.0</th>
      <td>-4.778777</td>
      <td>-4.801472</td>
      <td>-4.823926</td>
      <td>-4.846139</td>
      <td>-4.868108</td>
      <td>-4.889820</td>
      <td>-4.911257</td>
      <td>-4.932405</td>
      <td>-4.953249</td>
      <td>-4.973776</td>
      <td>...</td>
      <td>41.477796</td>
      <td>41.484073</td>
      <td>41.490365</td>
      <td>41.496673</td>
      <td>41.502995</td>
      <td>41.509330</td>
      <td>41.515677</td>
      <td>41.522036</td>
      <td>41.528405</td>
      <td>41.534784</td>
    </tr>
    <tr>
      <th>3.1</th>
      <td>-4.737781</td>
      <td>-4.760513</td>
      <td>-4.783006</td>
      <td>-4.805258</td>
      <td>-4.827267</td>
      <td>-4.849019</td>
      <td>-4.870497</td>
      <td>-4.891687</td>
      <td>-4.912574</td>
      <td>-4.933144</td>
      <td>...</td>
      <td>41.304409</td>
      <td>41.310674</td>
      <td>41.316956</td>
      <td>41.323253</td>
      <td>41.329564</td>
      <td>41.335888</td>
      <td>41.342225</td>
      <td>41.348573</td>
      <td>41.354931</td>
      <td>41.361298</td>
    </tr>
    <tr>
      <th>3.2</th>
      <td>-4.696562</td>
      <td>-4.719330</td>
      <td>-4.741860</td>
      <td>-4.764150</td>
      <td>-4.786198</td>
      <td>-4.807989</td>
      <td>-4.829508</td>
      <td>-4.850738</td>
      <td>-4.871666</td>
      <td>-4.892278</td>
      <td>...</td>
      <td>41.131211</td>
      <td>41.137466</td>
      <td>41.143737</td>
      <td>41.150023</td>
      <td>41.156324</td>
      <td>41.162637</td>
      <td>41.168963</td>
      <td>41.175300</td>
      <td>41.181647</td>
      <td>41.188003</td>
    </tr>
    <tr>
      <th>3.3</th>
      <td>-4.655069</td>
      <td>-4.677873</td>
      <td>-4.700438</td>
      <td>-4.722765</td>
      <td>-4.744850</td>
      <td>-4.766679</td>
      <td>-4.788237</td>
      <td>-4.809507</td>
      <td>-4.830475</td>
      <td>-4.851128</td>
      <td>...</td>
      <td>40.958244</td>
      <td>40.964488</td>
      <td>40.970749</td>
      <td>40.977024</td>
      <td>40.983314</td>
      <td>40.989616</td>
      <td>40.995931</td>
      <td>41.002257</td>
      <td>41.008594</td>
      <td>41.014939</td>
    </tr>
    <tr>
      <th>3.4</th>
      <td>-4.613256</td>
      <td>-4.636093</td>
      <td>-4.658693</td>
      <td>-4.681055</td>
      <td>-4.703175</td>
      <td>-4.725041</td>
      <td>-4.746636</td>
      <td>-4.767944</td>
      <td>-4.788951</td>
      <td>-4.809643</td>
      <td>...</td>
      <td>40.785545</td>
      <td>40.791779</td>
      <td>40.798029</td>
      <td>40.804294</td>
      <td>40.810573</td>
      <td>40.816865</td>
      <td>40.823169</td>
      <td>40.829484</td>
      <td>40.835810</td>
      <td>40.842145</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 4383 columns</p>
</div>



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
    
