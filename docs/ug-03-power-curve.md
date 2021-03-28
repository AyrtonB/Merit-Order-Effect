# Quantile LOWESS for Data Cleaning



[![Binder](https://notebooks.gesis.org/binder/badge_logo.svg)](https://notebooks.gesis.org/binder/v2/gh/AyrtonB/Merit-Order-Effect/main?filepath=nbs%2Fug-03-power-curve.ipynb)

In this notebook we'll look at how we can use our LOWESS methods to process wind turbine SCADA data and fit a clean power curve that doesn't include under-powered periods.

<br>

### Imports

```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from moepy import lowess, eda
```

<br>

### Loading Data

We'll start by loading the SCADA data in

```python
df_SCADA = pd.read_csv('../data/lowess_examples/turbine_SCADA.csv')

df_SCADA.head(3)
```




|   Unnamed: 0 | UTC                 |   T01_active_power |   T01_blade_pitch |   T01_generator_speed |   T01_wind_speed |   T02_active_power |   T02_blade_pitch |   T02_generator_speed |   T02_wind_speed |   T03_active_power | ...   |   T09_theoretical_power |   T10_theoretical_power |   T11_theoretical_power |   T12_theoretical_power |   T13_theoretical_power |   T14_theoretical_power |   T15_theoretical_power |   T16_theoretical_power |   pc_error |   wind_angle_rad |
|-------------:|:--------------------|-------------------:|------------------:|----------------------:|-----------------:|-------------------:|------------------:|----------------------:|-----------------:|-------------------:|:------|------------------------:|------------------------:|------------------------:|------------------------:|------------------------:|------------------------:|------------------------:|------------------------:|-----------:|-----------------:|
|            0 | 2017-09-30 18:26:40 |                 56 |              -0.4 |                   573 |             4.51 |                 -4 |               5.2 |                   682 |             2.21 |                -41 | ...   |                     163 |                       0 |                   23.32 |                  101.26 |                  121.42 |                   32.33 |                   54.88 |                   93.42 |    838.636 |              nan |
|            1 | 2017-09-30 18:56:40 |                 56 |              -0.4 |                   573 |             4.51 |                 -4 |               5.2 |                   682 |             2.21 |                -41 | ...   |                     163 |                       0 |                   23.32 |                  101.26 |                  121.42 |                   32.33 |                   54.88 |                   93.42 |    838.636 |              nan |
|            2 | 2017-09-30 19:26:40 |                 56 |              -0.4 |                   573 |             4.51 |                 -4 |               5.2 |                   682 |             2.21 |                -41 | ...   |                     163 |                       0 |                   23.32 |                  101.26 |                  121.42 |                   32.33 |                   54.88 |                   93.42 |    838.636 |              nan |</div>



<br>

We'll extract data for a single turbine

```python
filter_for_turbine = lambda df, turbine=1: df[['UTC', f'T{str(turbine).zfill(2)}_active_power', f'T{str(turbine).zfill(2)}_wind_speed', 'wind_angle_rad']].pipe(lambda df: df.rename(columns=dict(zip(df.columns, df.columns.str.replace(f'T{str(turbine).zfill(2)}_', '')))))

df_turbine = filter_for_turbine(df_SCADA, 15).dropna()

df_turbine.head()
```




|   Unnamed: 0 | UTC                 |   active_power |   wind_speed |   wind_angle_rad |
|-------------:|:--------------------|---------------:|-------------:|-----------------:|
|           12 | 2017-10-01 00:26:40 |           68   |      4.39667 |        -1.03557  |
|           13 | 2017-10-01 00:56:40 |          112.5 |      3.69    |        -0.969684 |
|           14 | 2017-10-01 01:26:40 |          238   |      4.67    |         0.395575 |
|           15 | 2017-10-01 01:56:40 |          494   |      7.14    |        -1.46098  |
|           16 | 2017-10-01 02:26:40 |          562   |      5.37    |         0.493009 |</div>



<br>

### Power Curve Cleaning

We'll then try and fit a Lowess estimate for the power curve.

```python
%%time

x = df_turbine['wind_speed'].values
y = df_turbine['active_power'].values

lowess_model = lowess.Lowess()
lowess_model.fit(x, y, frac=0.2, num_fits=100)

x_pred = np.linspace(0, 25, 101)
y_pred = lowess_model.predict(x_pred)

# Plotting
plt.plot(x_pred, y_pred, '--', label='Robust LOESS', color='k', zorder=3)
plt.scatter(x, y, label='With Noise', color='C1', s=1, zorder=1)
plt.legend(frameon=False)
```

    c:\users\ayrto\desktop\phd\analysis\merit-order-effect\moepy\lowess.py:145: FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.
    To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.
      betas = np.linalg.lstsq(A, b)[0]
    

    Wall time: 585 ms
    




    <matplotlib.legend.Legend at 0x1d5af4428e0>




![png](./img/nbs/output_7_3.png)


<br>

Unfortunately the fit is thrown by the large number of occurences where the farm is under-powered or set to output 0, we want to remove these so that we can estimate the 'standard' power curve. We'll create a quantile Lowess fit to see if that helps us understand the data any better.

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
<span class="Time-label">[02:08<00:03, 3.12s/it]</span></div>





|    x |       p97 |      p95 |      p92 |      p90 |      p88 |      p85 |      p83 |      p80 |      p78 |      p76 | ...   |      p23 |      p21 |      p19 |      p16 |      p14 |      p12 |       p9 |       p7 |       p4 |       p2 |
|-----:|----------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|:------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| 0.25 | -31.8607  | -36.669  | -49.5917 | -50.8011 | -54.8979 | -62.4346 | -65.6697 | -68.2331 | -68.9789 | -69.7274 | ...   | -79.5962 | -74.3463 | -69.0825 | -62.7053 | -58.362  | -53.9233 | -50.0939 | -49.134  | -50.205  | -51.815  |
| 0.33 | -25.204   | -31.2919 | -44.4735 | -46.1838 | -50.4993 | -58.1508 | -61.4824 | -64.1785 | -65.0755 | -65.9571 | ...   | -78.2436 | -73.3799 | -68.4272 | -62.3658 | -58.1436 | -53.8781 | -50.0808 | -49.0606 | -49.9814 | -51.5724 |
| 0.41 | -18.6577  | -25.9899 | -39.441  | -41.6317 | -46.1561 | -53.9169 | -57.3385 | -60.163  | -61.2086 | -62.2258 | ...   | -76.897  | -72.4143 | -67.7797 | -62.0494 | -57.9512 | -53.8512 | -50.0791 | -49.0013 | -49.7576 | -51.3317 |
| 0.49 | -12.1758  | -20.7247 | -34.4592 | -37.1132 | -41.8401 | -49.7063 | -53.2149 | -56.1647 | -57.3571 | -58.5129 | ...   | -75.5433 | -71.4351 | -67.1239 | -61.7374 | -57.7682 | -53.8289 | -50.0794 | -48.9493 | -49.5327 | -51.0921 |
| 0.57 |  -5.71114 | -15.4572 | -29.4922 | -32.596  | -37.5225 | -45.4924 | -49.0885 | -52.1612 | -53.4996 | -54.7973 | ...   | -74.1691 | -70.4282 | -66.4441 | -61.4111 | -57.5778 | -53.7971 | -50.0722 | -48.8973 | -49.3056 | -50.853  |</div>



<br>

Plotting these we can see an interesting relationship, where for many of the lower quantiles the estimate peaks then quickly drops to around 0. It is the area below and to the right of these peaks that we want to remove from our power curve estimate.

```python
fig, ax = plt.subplots(dpi=150)

ax.scatter(x, y, s=0.1, color='k', alpha=1)
df_quantiles.plot(cmap='viridis', legend=False, ax=ax)

eda.hide_spines(ax)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Active Power (MW)')
ax.set_xlim(0, 26)
ax.set_ylim(0)
```




    (0.0, 2715.0737814101485)




![png](./img/nbs/output_11_1.png)


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
    ax.set_ylabel('Active Power (MW)')
```


![png](./img/nbs/output_15_0.png)


<br>

We're now ready to make our power curve lowess estimate again

```python
%%time

lowess_model = lowess.Lowess()
lowess_model.fit(cleaned_x, cleaned_y, frac=0.2, num_fits=100)

x_pred = np.linspace(0, 25, 101)
y_pred = lowess_model.predict(x_pred)

# Plotting
fig, ax = plt.subplots(dpi=150)

ax.plot(x_pred, y_pred, '--', label='Robust LOESS', color='k', zorder=3)
ax.scatter(cleaned_x, cleaned_y, label='Observed', color='C1', s=0.5, zorder=1)

ax.legend(frameon=False)
eda.hide_spines(ax)
ax.set_xlim(0)
ax.set_ylim(0)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Active Power (MW)')
```

    c:\users\ayrto\desktop\phd\analysis\merit-order-effect\moepy\lowess.py:145: FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.
    To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.
      betas = np.linalg.lstsq(A, b)[0]
    

    Wall time: 550 ms
    




    Text(0, 0.5, 'Active Power (MW)')




![png](./img/nbs/output_17_3.png)


<br>

Potential areas for future exploration:
* Clip y to something like 0.01, just needs to be marginally above 0
* Should check what happens to the lower part though, because of frac the neg values may be helping
* What happens if I fit for data where power<2250 and speed<14
* Could then have a seperate distribution fit for the period after
* Could then use weights to transition between them
* Inspecting the active power spikes for the removed values should identify set-points
