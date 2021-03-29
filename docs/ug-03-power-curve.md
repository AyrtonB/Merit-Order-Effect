# Power Curves



[![Binder](https://notebooks.gesis.org/binder/badge_logo.svg)](https://notebooks.gesis.org/binder/v2/gh/AyrtonB/Merit-Order-Effect/main?filepath=nbs%2Fug-03-power-curve.ipynb)

In this notebook we'll look at how we can use our LOWESS methods to first fit a power curve for a wind turbine, then estimate the uncertainty in our results. We'll then utilise a feature of the quantile LOWESS fits to demonstrate how these techniques can also be used for cleaning raw turbine data.

<br>

### Imports

```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from moepy import lowess, eda
```

<br>

### Power Curve Fitting

We'll start by loading in some clean turbine output and wind speed data, this was sourced from the [Power Curve Working Group Analysis](https://github.com/PCWG/PCWG) example data repository.

```python
df_pcwga = (pd
            .read_csv('../data/lowess_examples/turbine_power_wind_speed_clean.csv')
            [['TimeStamp', 'Turbine Wind Speed Mean', 'Turbine Power']]
            .replace(-99.99, np.nan)
            .dropna()
           )

df_pcwga.head()
```




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
      <th>TimeStamp</th>
      <th>Turbine Wind Speed Mean</th>
      <th>Turbine Power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>07/10/2011 12:50</td>
      <td>15.510002</td>
      <td>1996.910019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>07/10/2011 13:00</td>
      <td>15.710131</td>
      <td>1987.739944</td>
    </tr>
    <tr>
      <th>2</th>
      <td>07/10/2011 13:10</td>
      <td>16.670750</td>
      <td>1991.900040</td>
    </tr>
    <tr>
      <th>3</th>
      <td>07/10/2011 13:20</td>
      <td>15.209808</td>
      <td>1987.700012</td>
    </tr>
    <tr>
      <th>4</th>
      <td>07/10/2011 13:30</td>
      <td>15.439957</td>
      <td>1991.030045</td>
    </tr>
  </tbody>
</table>
</div>



<br>

We'll then fit a standard LOWESS and visualise the results

```python
%%time

x = df_pcwga['Turbine Wind Speed Mean'].values
y = df_pcwga['Turbine Power'].values

lowess_model = lowess.Lowess()
lowess_model.fit(x, y, frac=0.2, num_fits=100)

x_pred = np.linspace(0, 25, 101)
y_pred = lowess_model.predict(x_pred)

# Plotting
fig, ax = plt.subplots(dpi=150)

ax.plot(x_pred, y_pred, label='Robust LOWESS', color='r', linewidth=1)
ax.scatter(x, y, label='Observations', s=0.5, color='k', linewidth=0, alpha=1)

eda.hide_spines(ax)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Power Output (MW)')
ax.set_xlim(0, 26)
ax.set_ylim(-25)

lgnd = ax.legend(frameon=False) 
lgnd.legendHandles[1]._sizes = [10]
lgnd.legendHandles[1].set_alpha(1)
```

    Wall time: 512 ms
    


![png](./img/nbs/ug-03-power-curve_cell_6_output_1.png)


<br>

This looks good but we can do more. In this next step we'll estimate the upper and lower quantiles of the power curve fit that represent a prediction interval of 68%.

```python
# Estimating the quantiles
df_quantiles = lowess.quantile_model(x, y, frac=0.2, qs=[0.16, 0.84], num_fits=40)

# Cleaning names and sorting for plotting
df_quantiles.columns = [f'p{int(col*100)}' for col in df_quantiles.columns]
df_quantiles = df_quantiles[df_quantiles.columns[::-1]]

df_quantiles.head()
```


<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:0; max-width:15ex; vertical-align:middle; text-align:right"></span>
<progress style="width:60ex" max="2" value="2" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">2/2</span>
<span class="Time-label">[00:06<00:03, 3.19s/it]</span></div>





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
      <th>p84</th>
      <th>p16</th>
    </tr>
    <tr>
      <th>x</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.323938</th>
      <td>-16.256912</td>
      <td>-8.347957</td>
    </tr>
    <tr>
      <th>0.463498</th>
      <td>-14.471922</td>
      <td>-8.538979</td>
    </tr>
    <tr>
      <th>0.475784</th>
      <td>-14.315327</td>
      <td>-8.556336</td>
    </tr>
    <tr>
      <th>0.503802</th>
      <td>-13.957291</td>
      <td>-8.596028</td>
    </tr>
    <tr>
      <th>0.546645</th>
      <td>-13.405900</td>
      <td>-8.656773</td>
    </tr>
  </tbody>
</table>
</div>



<br>

We'll visualise this fit within the domain where the quantiles do not cross

```python
valid_pred_intvl_idx = (df_quantiles['p84']-df_quantiles['p16']).pipe(lambda s: s[s>0]).index

# Plotting
fig, ax = plt.subplots(dpi=150)

ax.scatter(x, y, s=0.5, color='k', linewidth=0, alpha=1, label='Observations')
ax.fill_between(valid_pred_intvl_idx, df_quantiles.loc[valid_pred_intvl_idx, 'p16'], df_quantiles.loc[valid_pred_intvl_idx, 'p84'], color='r', alpha=0.25, label='68% Prediction Interval')

eda.hide_spines(ax)
ax.legend(frameon=False)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Power Output (MW)')
ax.set_xlim(0, 25)
ax.set_ylim(0)

lgnd = ax.legend(frameon=False) 
lgnd.legendHandles[0]._sizes = [20]
lgnd.legendHandles[0].set_alpha(1)
```


![png](./img/nbs/ug-03-power-curve_cell_10_output_0.png)


<br>

With the prediction interval we've looked at the likely range of power output for given wind speeds, but what if instead of the range of the underlying values we wanted to know the range in our estimate of the average power curve? For this we can use confidence intervals, which express the certainty we have in the particular statistical parameter we're calculating.

In order to estimate this uncertainty we'll first bootstrap our model.

```python
df_bootstrap = lowess.bootstrap_model(x, y, num_runs=500, frac=0.2, num_fits=30)

df_bootstrap.head()
```


<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:0; max-width:15ex; vertical-align:middle; text-align:right"></span>
<progress style="width:60ex" max="500" value="500" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">500/500</span>
<span class="Time-label">[00:55<00:00, 0.11s/it]</span></div>





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
      <th>bootstrap_run</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>490</th>
      <th>491</th>
      <th>492</th>
      <th>493</th>
      <th>494</th>
      <th>495</th>
      <th>496</th>
      <th>497</th>
      <th>498</th>
      <th>499</th>
    </tr>
    <tr>
      <th>x</th>
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
      <th>15.510002</th>
      <td>1977.775109</td>
      <td>1978.050462</td>
      <td>1978.950885</td>
      <td>1978.325069</td>
      <td>1979.434901</td>
      <td>1978.328213</td>
      <td>1978.780345</td>
      <td>1979.793830</td>
      <td>1979.469618</td>
      <td>1979.000728</td>
      <td>...</td>
      <td>1978.142412</td>
      <td>1978.496950</td>
      <td>1977.232973</td>
      <td>1977.661125</td>
      <td>1976.368042</td>
      <td>1976.278425</td>
      <td>1978.609513</td>
      <td>1978.842863</td>
      <td>1977.938420</td>
      <td>1977.216674</td>
    </tr>
    <tr>
      <th>15.710131</th>
      <td>1978.479303</td>
      <td>1978.599593</td>
      <td>1979.519515</td>
      <td>1979.054431</td>
      <td>1979.956749</td>
      <td>1979.059946</td>
      <td>1979.392922</td>
      <td>1980.336131</td>
      <td>1980.101091</td>
      <td>1979.740611</td>
      <td>...</td>
      <td>1978.924262</td>
      <td>1979.136631</td>
      <td>1977.816212</td>
      <td>1978.381958</td>
      <td>1977.141258</td>
      <td>1977.140990</td>
      <td>1979.447743</td>
      <td>1979.563367</td>
      <td>1978.607373</td>
      <td>1977.958401</td>
    </tr>
    <tr>
      <th>16.670750</th>
      <td>1981.239370</td>
      <td>1981.170351</td>
      <td>1981.832427</td>
      <td>1981.567940</td>
      <td>1981.869492</td>
      <td>1981.946969</td>
      <td>1981.611866</td>
      <td>1982.227839</td>
      <td>1982.299750</td>
      <td>1982.309773</td>
      <td>...</td>
      <td>1981.984375</td>
      <td>1981.560051</td>
      <td>1980.129383</td>
      <td>1981.419650</td>
      <td>1980.297856</td>
      <td>1980.601351</td>
      <td>1982.854191</td>
      <td>1981.927171</td>
      <td>1981.146967</td>
      <td>1980.829410</td>
    </tr>
    <tr>
      <th>15.209808</th>
      <td>1976.429380</td>
      <td>1977.042706</td>
      <td>1977.795060</td>
      <td>1977.101172</td>
      <td>1978.407717</td>
      <td>1976.897971</td>
      <td>1977.580780</td>
      <td>1978.667617</td>
      <td>1978.209257</td>
      <td>1977.578692</td>
      <td>...</td>
      <td>1976.642255</td>
      <td>1977.265530</td>
      <td>1976.052214</td>
      <td>1976.449853</td>
      <td>1974.944704</td>
      <td>1974.572961</td>
      <td>1977.017120</td>
      <td>1977.517115</td>
      <td>1976.609814</td>
      <td>1975.824605</td>
    </tr>
    <tr>
      <th>15.439957</th>
      <td>1977.491798</td>
      <td>1977.844801</td>
      <td>1978.720474</td>
      <td>1978.031565</td>
      <td>1979.210071</td>
      <td>1978.029925</td>
      <td>1978.520519</td>
      <td>1979.564238</td>
      <td>1979.202501</td>
      <td>1978.693672</td>
      <td>...</td>
      <td>1977.827458</td>
      <td>1978.231698</td>
      <td>1976.990201</td>
      <td>1977.402475</td>
      <td>1976.059118</td>
      <td>1975.929541</td>
      <td>1978.277731</td>
      <td>1978.537433</td>
      <td>1977.663896</td>
      <td>1976.918336</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 500 columns</p>
</div>



<br>

From the bootstrapped results we can then extract the confidence intervals, in our case we'll look at the range covering 95% of our estimates.

```python
df_conf_intvl = lowess.get_confidence_interval(df_bootstrap, conf_pct=0.95)

# Plotting
fig, ax = plt.subplots(dpi=150)

ax.scatter(x, y, s=0.5, color='k', linewidth=0, alpha=1, label='Observations')
ax.fill_between(df_conf_intvl.index, df_conf_intvl['min'], df_conf_intvl['max'], color='r', alpha=1, label='95% Confidence')

eda.hide_spines(ax)
ax.legend(frameon=False)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Power Output (MW)')
ax.set_xlim(0, 25)
ax.set_ylim(0)

lgnd = ax.legend(frameon=False) 
lgnd.legendHandles[0]._sizes = [20]
lgnd.legendHandles[0].set_alpha(1)
```


![png](./img/nbs/ug-03-power-curve_cell_14_output_0.png)


<br>

We'll now visualise how the width of the confidence interval changes with wind speed. Interestingly the two troughs in the confidence interval width appear to correspond to the cut-in and start of the rated power wind speeds.

```python
fig, ax = plt.subplots(dpi=150)

df_conf_intvl.diff(axis=1)['max'].plot(ax=ax, color='r', linewidth=1)

eda.hide_spines(ax)
ax.legend(frameon=False)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('95% Confidence\nInterval Width (MW)')
ax.set_xlim(0, 25)
ax.set_ylim(0)
ax.get_legend().remove()
```


![png](./img/nbs/ug-03-power-curve_cell_16_output_0.png)


<br>

### Power Curve Cleaning

We'll start by loading the raw turbine wind speed and output data in

```python
df_raw_pc = pd.read_csv('../data/lowess_examples/turbine_power_wind_speed_raw.csv')

df_raw_pc.head(3)
```




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
      <th>UTC</th>
      <th>active_power</th>
      <th>wind_speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30/09/2017 18:26</td>
      <td>14.0</td>
      <td>4.02</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30/09/2017 18:56</td>
      <td>14.0</td>
      <td>4.02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30/09/2017 19:26</td>
      <td>14.0</td>
      <td>4.02</td>
    </tr>
  </tbody>
</table>
</div>



<br>

We'll then try and fit a LOWESS estimate for the power curve.

```python
%%time

x = df_raw_pc['wind_speed'].values
y = df_raw_pc['active_power'].values

lowess_model = lowess.Lowess()
lowess_model.fit(x, y, frac=0.2, num_fits=100)

x_pred = np.linspace(0, 25, 101)
y_pred = lowess_model.predict(x_pred)

# Plotting
fig, ax = plt.subplots(dpi=150)

ax.plot(x_pred, y_pred, label='Robust LOWESS', color='r', linewidth=1)
ax.scatter(x, y, label='Observations', s=0.5, color='k', linewidth=0, alpha=1)

eda.hide_spines(ax)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Power Output (MW)')
ax.set_xlim(0, 26)
ax.set_ylim(-25)

lgnd = ax.legend(frameon=False) 
lgnd.legendHandles[1]._sizes = [10]
lgnd.legendHandles[1].set_alpha(1)
```

    Wall time: 788 ms
    


![png](./img/nbs/ug-03-power-curve_cell_20_output_1.png)


<br>

Unfortunately the fit is thrown by the large number of occurences where the farm is under-powered or set to output 0, we want to remove these so that we can estimate the 'standard' power curve. We'll create a quantile LOWESS fit to see if that helps us understand the data any better.

```python
# Estimating the quantiles
df_quantiles = lowess.quantile_model(x, y, frac=0.2, qs=np.linspace(0.025, 0.975, 41), num_fits=40)

# Cleaning names and sorting for plotting
df_quantiles.columns = [f'p{int(col*100)}' for col in df_quantiles.columns]
df_quantiles = df_quantiles[df_quantiles.columns[::-1]]

df_quantiles.head()
```


<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:0; max-width:15ex; vertical-align:middle; text-align:right"></span>
<progress style="width:60ex" max="41" value="41" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">41/41</span>
<span class="Time-label">[02:52<00:05, 4.20s/it]</span></div>





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
      <th>p97</th>
      <th>p95</th>
      <th>p92</th>
      <th>p90</th>
      <th>p88</th>
      <th>p85</th>
      <th>p83</th>
      <th>p80</th>
      <th>p78</th>
      <th>p76</th>
      <th>...</th>
      <th>p23</th>
      <th>p21</th>
      <th>p19</th>
      <th>p16</th>
      <th>p14</th>
      <th>p12</th>
      <th>p9</th>
      <th>p7</th>
      <th>p4</th>
      <th>p2</th>
    </tr>
    <tr>
      <th>x</th>
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
      <th>0.25</th>
      <td>-31.138943</td>
      <td>-36.063656</td>
      <td>-48.787512</td>
      <td>-50.225155</td>
      <td>-54.375248</td>
      <td>-61.881759</td>
      <td>-64.926254</td>
      <td>-67.370517</td>
      <td>-68.176035</td>
      <td>-69.032379</td>
      <td>...</td>
      <td>-78.788130</td>
      <td>-73.944711</td>
      <td>-69.321976</td>
      <td>-63.193786</td>
      <td>-58.940816</td>
      <td>-54.033732</td>
      <td>-50.657819</td>
      <td>-49.236591</td>
      <td>-50.801038</td>
      <td>-51.824759</td>
    </tr>
    <tr>
      <th>0.33</th>
      <td>-24.413665</td>
      <td>-30.610793</td>
      <td>-43.604414</td>
      <td>-45.520463</td>
      <td>-49.912984</td>
      <td>-57.545203</td>
      <td>-60.729165</td>
      <td>-63.275040</td>
      <td>-64.235407</td>
      <td>-65.233799</td>
      <td>...</td>
      <td>-77.391861</td>
      <td>-72.894614</td>
      <td>-68.579404</td>
      <td>-62.740178</td>
      <td>-58.680432</td>
      <td>-53.961285</td>
      <td>-50.608743</td>
      <td>-49.144467</td>
      <td>-50.546003</td>
      <td>-51.578860</td>
    </tr>
    <tr>
      <th>0.41</th>
      <td>-17.837829</td>
      <td>-25.259973</td>
      <td>-38.527423</td>
      <td>-40.896936</td>
      <td>-45.521473</td>
      <td>-53.274741</td>
      <td>-56.592850</td>
      <td>-59.231692</td>
      <td>-60.344065</td>
      <td>-61.487147</td>
      <td>...</td>
      <td>-76.014021</td>
      <td>-71.858512</td>
      <td>-67.852225</td>
      <td>-62.307253</td>
      <td>-58.453376</td>
      <td>-53.913426</td>
      <td>-50.574612</td>
      <td>-49.070218</td>
      <td>-50.291389</td>
      <td>-51.335279</td>
    </tr>
    <tr>
      <th>0.49</th>
      <td>-11.365853</td>
      <td>-19.974835</td>
      <td>-33.523677</td>
      <td>-36.325733</td>
      <td>-41.174405</td>
      <td>-49.045466</td>
      <td>-52.494676</td>
      <td>-55.220039</td>
      <td>-56.482108</td>
      <td>-57.772743</td>
      <td>...</td>
      <td>-74.641947</td>
      <td>-70.823212</td>
      <td>-67.126139</td>
      <td>-61.880349</td>
      <td>-58.243593</td>
      <td>-53.876901</td>
      <td>-50.546571</td>
      <td>-49.007175</td>
      <td>-50.036505</td>
      <td>-51.093366</td>
    </tr>
    <tr>
      <th>0.57</th>
      <td>-4.947601</td>
      <td>-14.715394</td>
      <td>-28.557015</td>
      <td>-31.775095</td>
      <td>-36.842879</td>
      <td>-44.830165</td>
      <td>-48.409769</td>
      <td>-51.217707</td>
      <td>-52.627811</td>
      <td>-54.069014</td>
      <td>...</td>
      <td>-73.261943</td>
      <td>-69.774489</td>
      <td>-66.385850</td>
      <td>-61.443766</td>
      <td>-58.033785</td>
      <td>-53.837458</td>
      <td>-50.515147</td>
      <td>-48.948112</td>
      <td>-49.780591</td>
      <td>-50.852403</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 41 columns</p>
</div>



<br>

Plotting these we can see an interesting relationship, where for many of the lower quantiles the estimate peaks then quickly drops to around 0. It is the area below and to the right of these peaks that we want to remove from our power curve estimate.

```python
fig, ax = plt.subplots(dpi=150)

ax.scatter(x, y, s=0.1, color='k', alpha=1)
df_quantiles.plot(cmap='viridis', legend=False, ax=ax)

eda.hide_spines(ax)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Power Output (MW)')
ax.set_xlim(0, 26)
ax.set_ylim(0)
```




    (0.0, 2557.302509884304)




![png](./img/nbs/ug-03-power-curve_cell_24_output_1.png)


<br>

We'll first identify the ratio between the peak value and the final value (above speeds of 25 m/s). We'll then remove points after the peak where the ratio between their peak and final values exceeds a defined threshold (in our case 2).

N.b. there's probably a much nicer way to do this where the for loop isn't needed, this would be handy to implement as it would be good to have a vector containing all of the indexes that have been removed.

```python
exceeded_quantiles = ((df_quantiles.max()/df_quantiles.iloc[-1].clip(0.1) > 2)
                      .replace(False, np.nan)
                      .dropna()
                      .index
                     )

s_maxs = df_quantiles[exceeded_quantiles].max()
s_idxmaxs = df_quantiles[exceeded_quantiles].idxmax()

cleaned_x = x
cleaned_y = y

for exceeded_quantile in exceeded_quantiles:
    min_x = s_idxmaxs[exceeded_quantile]
    max_y = s_maxs[exceeded_quantile]
    
    idxs_to_remove = (cleaned_x > min_x) & (cleaned_y < max_y)

    cleaned_x = cleaned_x[~idxs_to_remove]
    cleaned_y = cleaned_y[~idxs_to_remove]
```

<br>

Visualising the results we can clearly see that the unwanted periods have been removed

```python
fig, axs = plt.subplots(dpi=250, ncols=2, figsize=(10, 4))

axs[0].scatter(x, y, s=0.1, color='k', alpha=1)
axs[1].scatter(cleaned_x, cleaned_y, s=0.1, color='k', alpha=1)

axs[0].set_title('Original')
axs[1].set_title('Cleaned')

for ax in axs:
    eda.hide_spines(ax)
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Power Output (MW)')
```


![png](./img/nbs/ug-03-power-curve_cell_28_output_0.png)


<br>

We're now ready to make our power curve LOWESS estimate again

```python
%%time

lowess_model = lowess.Lowess()
lowess_model.fit(cleaned_x, cleaned_y, frac=0.2, num_fits=100)

x_pred = np.linspace(0, 25, 101)
y_pred = lowess_model.predict(x_pred)

# Plotting
fig, ax = plt.subplots(dpi=150)

ax.plot(x_pred, y_pred, label='Robust LOWESS', color='r', linewidth=1)
ax.scatter(cleaned_x, cleaned_y, label='Observations', s=0.5, color='k', linewidth=0, alpha=1)

eda.hide_spines(ax)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Power Output (MW)')
ax.set_xlim(0, 26)
ax.set_ylim(-25)

lgnd = ax.legend(frameon=False) 
lgnd.legendHandles[1]._sizes = [10]
lgnd.legendHandles[1].set_alpha(1)
```

    Wall time: 728 ms
    


![png](./img/nbs/ug-03-power-curve_cell_30_output_1.png)

