# Wind Power Curve Fitting & Cleaning



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




|   Unnamed: 0 | TimeStamp        |   Turbine Wind Speed Mean |   Turbine Power |
|-------------:|:-----------------|--------------------------:|----------------:|
|            0 | 07/10/2011 12:50 |                   15.51   |         1996.91 |
|            1 | 07/10/2011 13:00 |                   15.7101 |         1987.74 |
|            2 | 07/10/2011 13:10 |                   16.6708 |         1991.9  |
|            3 | 07/10/2011 13:20 |                   15.2098 |         1987.7  |
|            4 | 07/10/2011 13:30 |                   15.44   |         1991.03 |</div>



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





|        x |      p84 |      p16 |
|---------:|---------:|---------:|
| 0.323938 | -16.2569 | -8.34796 |
| 0.463498 | -14.4719 | -8.53898 |
| 0.475784 | -14.3153 | -8.55634 |
| 0.503802 | -13.9573 | -8.59603 |
| 0.546645 | -13.4059 | -8.65677 |</div>



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





|       x |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 | ...   |     490 |     491 |     492 |     493 |     494 |     495 |     496 |     497 |     498 |     499 |
|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|:------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
| 15.51   | 1977.78 | 1978.05 | 1978.95 | 1978.33 | 1979.43 | 1978.33 | 1978.78 | 1979.79 | 1979.47 | 1979    | ...   | 1978.14 | 1978.5  | 1977.23 | 1977.66 | 1976.37 | 1976.28 | 1978.61 | 1978.84 | 1977.94 | 1977.22 |
| 15.7101 | 1978.48 | 1978.6  | 1979.52 | 1979.05 | 1979.96 | 1979.06 | 1979.39 | 1980.34 | 1980.1  | 1979.74 | ...   | 1978.92 | 1979.14 | 1977.82 | 1978.38 | 1977.14 | 1977.14 | 1979.45 | 1979.56 | 1978.61 | 1977.96 |
| 16.6708 | 1981.24 | 1981.17 | 1981.83 | 1981.57 | 1981.87 | 1981.95 | 1981.61 | 1982.23 | 1982.3  | 1982.31 | ...   | 1981.98 | 1981.56 | 1980.13 | 1981.42 | 1980.3  | 1980.6  | 1982.85 | 1981.93 | 1981.15 | 1980.83 |
| 15.2098 | 1976.43 | 1977.04 | 1977.8  | 1977.1  | 1978.41 | 1976.9  | 1977.58 | 1978.67 | 1978.21 | 1977.58 | ...   | 1976.64 | 1977.27 | 1976.05 | 1976.45 | 1974.94 | 1974.57 | 1977.02 | 1977.52 | 1976.61 | 1975.82 |
| 15.44   | 1977.49 | 1977.84 | 1978.72 | 1978.03 | 1979.21 | 1978.03 | 1978.52 | 1979.56 | 1979.2  | 1978.69 | ...   | 1977.83 | 1978.23 | 1976.99 | 1977.4  | 1976.06 | 1975.93 | 1978.28 | 1978.54 | 1977.66 | 1976.92 |</div>



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


```python
# take the peak, then get the troughs on either side and print some info
# would be interesting to compare against these fits - https://www.hindawi.com/journals/jen/2016/8519785/tab1/
```

<br>

### Power Curve Cleaning

We'll start by loading the raw turbine wind speed and output data in

```python
df_raw_pc = pd.read_csv('../data/lowess_examples/turbine_power_wind_speed_raw.csv')

df_raw_pc.head(3)
```




|   Unnamed: 0 | UTC              |   active_power |   wind_speed |
|-------------:|:-----------------|---------------:|-------------:|
|            0 | 30/09/2017 18:26 |             14 |         4.02 |
|            1 | 30/09/2017 18:56 |             14 |         4.02 |
|            2 | 30/09/2017 19:26 |             14 |         4.02 |</div>



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
    


![png](./img/nbs/ug-03-power-curve_cell_21_output_1.png)


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





|    x |      p97 |      p95 |      p92 |      p90 |      p88 |      p85 |      p83 |      p80 |      p78 |      p76 | ...   |      p23 |      p21 |      p19 |      p16 |      p14 |      p12 |       p9 |       p7 |       p4 |       p2 |
|-----:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|:------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| 0.25 | -31.1389 | -36.0637 | -48.7875 | -50.2252 | -54.3752 | -61.8818 | -64.9263 | -67.3705 | -68.176  | -69.0324 | ...   | -78.7881 | -73.9447 | -69.322  | -63.1938 | -58.9408 | -54.0337 | -50.6578 | -49.2366 | -50.801  | -51.8248 |
| 0.33 | -24.4137 | -30.6108 | -43.6044 | -45.5205 | -49.913  | -57.5452 | -60.7292 | -63.275  | -64.2354 | -65.2338 | ...   | -77.3919 | -72.8946 | -68.5794 | -62.7402 | -58.6804 | -53.9613 | -50.6087 | -49.1445 | -50.546  | -51.5789 |
| 0.41 | -17.8378 | -25.26   | -38.5274 | -40.8969 | -45.5215 | -53.2747 | -56.5928 | -59.2317 | -60.3441 | -61.4871 | ...   | -76.014  | -71.8585 | -67.8522 | -62.3073 | -58.4534 | -53.9134 | -50.5746 | -49.0702 | -50.2914 | -51.3353 |
| 0.49 | -11.3659 | -19.9748 | -33.5237 | -36.3257 | -41.1744 | -49.0455 | -52.4947 | -55.22   | -56.4821 | -57.7727 | ...   | -74.6419 | -70.8232 | -67.1261 | -61.8803 | -58.2436 | -53.8769 | -50.5466 | -49.0072 | -50.0365 | -51.0934 |
| 0.57 |  -4.9476 | -14.7154 | -28.557  | -31.7751 | -36.8429 | -44.8302 | -48.4098 | -51.2177 | -52.6278 | -54.069  | ...   | -73.2619 | -69.7745 | -66.3859 | -61.4438 | -58.0338 | -53.8375 | -50.5151 | -48.9481 | -49.7806 | -50.8524 |</div>



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




![png](./img/nbs/ug-03-power-curve_cell_25_output_1.png)


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


![png](./img/nbs/ug-03-power-curve_cell_29_output_0.png)


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
    


![png](./img/nbs/ug-03-power-curve_cell_31_output_1.png)

