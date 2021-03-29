# Estimation of Price Surfaces



[![Binder](https://notebooks.gesis.org/binder/badge_logo.svg)](https://notebooks.gesis.org/binder/v2/gh/AyrtonB/Merit-Order-Effect/main?filepath=nbs%2Fdev-04-price-surface-estimation.ipynb)

This notebook outlines how to specify different variants the model, then proceeds to fit them.

<br>

### Imports

```python
#exports
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import os
import pickle
import FEAutils as hlp
from ipypb import track

from moepy import lowess, eda
```

<br>

### User Inputs

```python
models_dir = '../data/models'
load_existing_model = True
```

<br>

### Loading & Cleaning Data

We'll start by loading in ...

```python
%%time

df_EI = eda.load_EI_df('../data/raw/electric_insights.csv')

df_EI.head()
```

    Wall time: 1.69 s
    




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
      <th>day_ahead_price</th>
      <th>SP</th>
      <th>imbalance_price</th>
      <th>valueSum</th>
      <th>temperature</th>
      <th>TCO2_per_h</th>
      <th>gCO2_per_kWh</th>
      <th>nuclear</th>
      <th>biomass</th>
      <th>coal</th>
      <th>...</th>
      <th>demand</th>
      <th>pumped_storage</th>
      <th>wind_onshore</th>
      <th>wind_offshore</th>
      <th>belgian</th>
      <th>dutch</th>
      <th>french</th>
      <th>ireland</th>
      <th>northern_ireland</th>
      <th>irish</th>
    </tr>
    <tr>
      <th>local_datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-01-01 00:00:00+00:00</th>
      <td>58.05</td>
      <td>1</td>
      <td>74.74</td>
      <td>74.74</td>
      <td>-0.6</td>
      <td>21278.0</td>
      <td>555.0</td>
      <td>6.973</td>
      <td>0.0</td>
      <td>17.650</td>
      <td>...</td>
      <td>38.329</td>
      <td>-0.404</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.977</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.161</td>
    </tr>
    <tr>
      <th>2009-01-01 00:30:00+00:00</th>
      <td>56.33</td>
      <td>2</td>
      <td>74.89</td>
      <td>74.89</td>
      <td>-0.6</td>
      <td>21442.0</td>
      <td>558.0</td>
      <td>6.968</td>
      <td>0.0</td>
      <td>17.770</td>
      <td>...</td>
      <td>38.461</td>
      <td>-0.527</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.977</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.160</td>
    </tr>
    <tr>
      <th>2009-01-01 01:00:00+00:00</th>
      <td>52.98</td>
      <td>3</td>
      <td>76.41</td>
      <td>76.41</td>
      <td>-0.6</td>
      <td>21614.0</td>
      <td>569.0</td>
      <td>6.970</td>
      <td>0.0</td>
      <td>18.070</td>
      <td>...</td>
      <td>37.986</td>
      <td>-1.018</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.977</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.160</td>
    </tr>
    <tr>
      <th>2009-01-01 01:30:00+00:00</th>
      <td>50.39</td>
      <td>4</td>
      <td>37.73</td>
      <td>37.73</td>
      <td>-0.6</td>
      <td>21320.0</td>
      <td>578.0</td>
      <td>6.969</td>
      <td>0.0</td>
      <td>18.022</td>
      <td>...</td>
      <td>36.864</td>
      <td>-1.269</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.746</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.160</td>
    </tr>
    <tr>
      <th>2009-01-01 02:00:00+00:00</th>
      <td>48.70</td>
      <td>5</td>
      <td>59.00</td>
      <td>59.00</td>
      <td>-0.6</td>
      <td>21160.0</td>
      <td>585.0</td>
      <td>6.960</td>
      <td>0.0</td>
      <td>17.998</td>
      <td>...</td>
      <td>36.180</td>
      <td>-1.566</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.730</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.160</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



<br>

... and cleaning the GB data

```python
df_EI_model = df_EI[['day_ahead_price', 'demand', 'solar', 'wind']].dropna()

s_demand = df_EI_model['demand']
s_price = df_EI_model['day_ahead_price']
s_dispatchable = df_EI_model['demand'] - df_EI_model[['solar', 'wind']].sum(axis=1)

# Plotting
fig, ax = plt.subplots(dpi=150)

ax.scatter(s_dispatchable['2010-09':'2011-03'], s_price['2010-09':'2011-03'], s=1)
ax.scatter(s_dispatchable['2020-03':'2020-09'], s_price['2020-03':'2020-09'], s=1)

hlp.hide_spines(ax)
ax.set_xlim(8, 60)
ax.set_ylim(-25, 100)
ax.set_xlabel('Demand - [Wind + Solar] (MW)')
ax.set_ylabel('Price (£/MWh)')
```




    Text(0, 0.5, 'Price (£/MWh)')




![png](./img/nbs/output_7_1.png)


<br>

As well as the DE data

```python
df_DE = eda.load_DE_df('../data/raw/energy_charts.csv', '../data/raw/ENTSOE_DE_price.csv')

df_DE_model = df_DE[['price', 'demand', 'Solar', 'Wind']].dropna()

s_DE_demand = df_DE_model['demand']
s_DE_price = df_DE_model['price']
s_DE_dispatchable = df_DE_model['demand'] - df_DE_model[['Solar', 'Wind']].sum(axis=1)

# Plotting
fig, ax = plt.subplots(dpi=150)

ax.scatter(s_DE_dispatchable['2015-09':'2016-03'], s_DE_price['2015-09':'2016-03'], s=1)
ax.scatter(s_DE_dispatchable['2020-03':'2020-09'], s_DE_price['2020-03':'2020-09'], s=1)

hlp.hide_spines(ax)
ax.set_xlim(8, 75)
ax.set_ylim(-25, 100)
ax.set_xlabel('Demand - [Wind + Solar] (MW)')
ax.set_ylabel('Price (£/MWh)')
```




    Text(0, 0.5, 'Price (£/MWh)')




![png](./img/nbs/output_9_1.png)


<br>

### Results Wrapper

We'll start defining each of the price models that we'll fit, using the `PicklableFunction` class to ensure that all of our models can be saved for later use.

```python
#exports
import copy
import types
import marshal

class PicklableFunction:
    """Provides a wrapper to ensure functions can be pickled"""
    def __init__(self, fun):
        self._fun = fun

    def __call__(self, *args, **kwargs):
        return self._fun(*args, **kwargs)

    def __getstate__(self):
        try:
            return pickle.dumps(self._fun)
        except Exception:
            return marshal.dumps((self._fun.__code__, self._fun.__name__))

    def __setstate__(self, state):
        try:
            self._fun = pickle.loads(state)
        except Exception:
            code, name = marshal.loads(state)
            self._fun = types.FunctionType(code, {}, name)
            
        return
          
def get_fit_kwarg_sets(qs=np.linspace(0.1, 0.9, 9)):
    """Helper to generate kwargs for the `fit` method of `Lowess`"""
    fit_kwarg_sets = [
        # quantile lowess
        { 
            'name': f'p{int(q*100)}',
            'lowess_kwargs': {'reg_func': PicklableFunction(lowess.calc_quant_reg_betas)},
            'q': q,
        }
        for q in qs

        # standard lowess
    ] + [{'name': 'average'}] 
    
    return fit_kwarg_sets
```

```python
model_definitions = {
    'DAM_price_GB': {
        'dt_idx': s_dispatchable.index,
        'x': s_dispatchable.values,
        'y': s_price.values,
        'reg_dates_start': '2009-01-01',
        'reg_dates_end': '2021-01-01',
        'reg_dates_freq': '13W', # 13 
        'frac': 0.3, 
        'num_fits': 31, # 31
        'dates_smoothing_value': 26, # 26
        'dates_smoothing_units': 'W',
        'fit_kwarg_sets': get_fit_kwarg_sets(qs=[0.16, 0.5, 0.84])
    },
    'DAM_price_demand_GB': {
        'dt_idx': s_demand.index,
        'x': s_demand.values,
        'y': s_price.values,
        'reg_dates_start': '2009-01-01',
        'reg_dates_end': '2021-01-01',
        'reg_dates_freq': '13W', # 13 
        'frac': 0.3, 
        'num_fits': 31, # 31
        'dates_smoothing_value': 26, # 26
        'dates_smoothing_units': 'W',
        'fit_kwarg_sets': get_fit_kwarg_sets(qs=[0.5])
    },
    'DAM_price_DE': {
        'dt_idx': s_DE_dispatchable.index,
        'x': s_DE_dispatchable.values,
        'y': s_DE_price.values,
        'reg_dates_start': '2015-01-04',
        'reg_dates_end': '2021-01-01',
        'reg_dates_freq': '13W', # 13 
        'frac': 0.3, 
        'num_fits': 31, # 31
        'dates_smoothing_value': 26, # 26
        'dates_smoothing_units': 'W',
        'fit_kwarg_sets': get_fit_kwarg_sets(qs=[0.16, 0.5, 0.84])
    },
    'DAM_price_demand_DE': {
        'dt_idx': s_DE_dispatchable.index,
        'x': s_DE_demand.values,
        'y': s_DE_price.values,
        'reg_dates_start': '2015-01-04',
        'reg_dates_end': '2021-01-01',
        'reg_dates_freq': '13W', # 13 
        'frac': 0.3, 
        'num_fits': 31, # 31
        'dates_smoothing_value': 26, # 26
        'dates_smoothing_units': 'W',
        'fit_kwarg_sets': get_fit_kwarg_sets(qs=[0.5])
    }
}
```

<br>

We'll now take these model definitions to fit and save them

```python
#exports
def fit_models(model_definitions, models_dir):
    """Fits LOWESS variants using the specified model definitions"""
    for model_parent_name, model_spec in model_definitions.items():
        for fit_kwarg_set in track(model_spec['fit_kwarg_sets'], label=model_parent_name):
            run_name = fit_kwarg_set.pop('name')
            model_name = f'{model_parent_name}_{run_name}'

            if f'{model_name}.pkl' not in os.listdir(models_dir):
                smooth_dates = lowess.SmoothDates()

                reg_dates = pd.date_range(
                    model_spec['reg_dates_start'], 
                    model_spec['reg_dates_end'], 
                    freq=model_spec['reg_dates_freq']
                )
                
                smooth_dates.fit(
                    model_spec['x'], 
                    model_spec['y'], 
                    dt_idx=model_spec['dt_idx'], 
                    reg_dates=reg_dates, 
                    frac=model_spec['frac'], 
                    threshold_value=model_spec['dates_smoothing_value'], 
                    threshold_units=model_spec['dates_smoothing_units'],
                    num_fits=model_spec['num_fits'], 
                    **fit_kwarg_set
                )
                
                model_fp = f'{models_dir}/{model_name}.pkl'
                pickle.dump(smooth_dates, open(model_fp, 'wb'))

                del smooth_dates
```

```python
fit_models(model_definitions, models_dir)
```


<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:15ex; max-width:15ex; vertical-align:middle; text-align:right">DAM_price_GB</span>
<progress style="width:45ex" max="4" value="4" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">4/4</span>
<span class="Time-label">[00:00<00:00, 0.00s/it]</span></div>



<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:15ex; max-width:15ex; vertical-align:middle; text-align:right">DAM_price_demand_GB</span>
<progress style="width:45ex" max="2" value="2" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">2/2</span>
<span class="Time-label">[00:00<00:00, 0.00s/it]</span></div>



<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:15ex; max-width:15ex; vertical-align:middle; text-align:right">DAM_price_DE</span>
<progress style="width:45ex" max="4" value="4" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">4/4</span>
<span class="Time-label">[00:00<00:00, 0.00s/it]</span></div>



<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:15ex; max-width:15ex; vertical-align:middle; text-align:right">DAM_price_demand_DE</span>
<progress style="width:45ex" max="2" value="2" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">2/2</span>
<span class="Time-label">[00:00<00:00, 0.00s/it]</span></div>


<br>

We'll load one of the models in

```python
%%time

if load_existing_model == True:
    smooth_dates = pickle.load(open(f'{models_dir}/DAM_price_GB_p50.pkl', 'rb'))
else:
    lowess_kwargs = {}
    reg_dates = pd.date_range('2009-01-01', '2021-01-01', freq='13W')

    smooth_dates = lowess.SmoothDates()
    smooth_dates.fit(s_dispatchable.values, s_price.values, dt_idx=s_dispatchable.index, 
                     reg_dates=reg_dates, frac=0.3, num_fits=31, threshold_value=26, lowess_kwargs=lowess_kwargs)
```

    Wall time: 2.7 s
    

<br>

And create a prediction surface using it

```python
%%time

x_pred = np.linspace(8, 60, 521)
dt_pred = pd.date_range('2009-01-01', '2021-01-01', freq='1W')

df_pred = smooth_dates.predict(x_pred=x_pred, dt_pred=dt_pred)

df_pred.head()
```

    Wall time: 346 ms
    




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
      <th>2009-01-04</th>
      <th>2009-01-11</th>
      <th>2009-01-18</th>
      <th>2009-01-25</th>
      <th>2009-02-01</th>
      <th>2009-02-08</th>
      <th>2009-02-15</th>
      <th>2009-02-22</th>
      <th>2009-03-01</th>
      <th>2009-03-08</th>
      <th>...</th>
      <th>2020-10-25</th>
      <th>2020-11-01</th>
      <th>2020-11-08</th>
      <th>2020-11-15</th>
      <th>2020-11-22</th>
      <th>2020-11-29</th>
      <th>2020-12-06</th>
      <th>2020-12-13</th>
      <th>2020-12-20</th>
      <th>2020-12-27</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8.0</th>
      <td>-7.660008</td>
      <td>-7.789268</td>
      <td>-7.910814</td>
      <td>-8.025717</td>
      <td>-8.134805</td>
      <td>-8.238751</td>
      <td>-8.338129</td>
      <td>-8.433449</td>
      <td>-8.525193</td>
      <td>-8.613820</td>
      <td>...</td>
      <td>10.235374</td>
      <td>10.292018</td>
      <td>10.347611</td>
      <td>10.402138</td>
      <td>10.455693</td>
      <td>10.508530</td>
      <td>10.561129</td>
      <td>10.614270</td>
      <td>10.669140</td>
      <td>10.727071</td>
    </tr>
    <tr>
      <th>8.1</th>
      <td>-7.467721</td>
      <td>-7.596370</td>
      <td>-7.717343</td>
      <td>-7.831705</td>
      <td>-7.940282</td>
      <td>-8.043744</td>
      <td>-8.142661</td>
      <td>-8.237542</td>
      <td>-8.328865</td>
      <td>-8.417088</td>
      <td>...</td>
      <td>10.442911</td>
      <td>10.499384</td>
      <td>10.554824</td>
      <td>10.609219</td>
      <td>10.662661</td>
      <td>10.715403</td>
      <td>10.767921</td>
      <td>10.820990</td>
      <td>10.875787</td>
      <td>10.933636</td>
    </tr>
    <tr>
      <th>8.2</th>
      <td>-7.275607</td>
      <td>-7.403641</td>
      <td>-7.524036</td>
      <td>-7.637854</td>
      <td>-7.745917</td>
      <td>-7.848890</td>
      <td>-7.947342</td>
      <td>-8.041780</td>
      <td>-8.132678</td>
      <td>-8.220493</td>
      <td>...</td>
      <td>10.650337</td>
      <td>10.706638</td>
      <td>10.761927</td>
      <td>10.816190</td>
      <td>10.869521</td>
      <td>10.922169</td>
      <td>10.974607</td>
      <td>11.027605</td>
      <td>11.082331</td>
      <td>11.140099</td>
    </tr>
    <tr>
      <th>8.3</th>
      <td>-7.083662</td>
      <td>-7.211075</td>
      <td>-7.330889</td>
      <td>-7.444158</td>
      <td>-7.551701</td>
      <td>-7.654183</td>
      <td>-7.752166</td>
      <td>-7.846157</td>
      <td>-7.936627</td>
      <td>-8.024030</td>
      <td>...</td>
      <td>10.857636</td>
      <td>10.913767</td>
      <td>10.968906</td>
      <td>11.023039</td>
      <td>11.076259</td>
      <td>11.128814</td>
      <td>11.181173</td>
      <td>11.234100</td>
      <td>11.288756</td>
      <td>11.346444</td>
    </tr>
    <tr>
      <th>8.4</th>
      <td>-6.891877</td>
      <td>-7.018666</td>
      <td>-7.137894</td>
      <td>-7.250611</td>
      <td>-7.357631</td>
      <td>-7.459617</td>
      <td>-7.557128</td>
      <td>-7.650668</td>
      <td>-7.740706</td>
      <td>-7.827694</td>
      <td>...</td>
      <td>11.064795</td>
      <td>11.120757</td>
      <td>11.175747</td>
      <td>11.229751</td>
      <td>11.282861</td>
      <td>11.335324</td>
      <td>11.387606</td>
      <td>11.440464</td>
      <td>11.495050</td>
      <td>11.552659</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 626 columns</p>
</div>


