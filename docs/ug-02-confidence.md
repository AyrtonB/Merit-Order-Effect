# Bootstrapped LOWESS for Confidence Intervals



This notebook outlines how to use the `moepy` library to generate confidence intervals around the LOWESS estimates, using the famous LIGO gravitationl wave data as an example. 

N.b. I have no expertise of signal processing in this particular context, this is merely an example of how LOWESS confidence intervals can be used to limit the domain of your prediction to where you have greatest certainty in it.

<br>

### Imports

```python
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from moepy import lowess, eda
```

<br>

### Loading Data

We'll start by loading the LIGO data in

```python
df_LIGO = pd.read_csv('../data/lowess_examples/LIGO.csv')

df_LIGO.head()
```




|   Unnamed: 0 |   frequency |          L1 |          H1 |   H1_smoothed |
|-------------:|------------:|------------:|------------:|--------------:|
|            0 |        0    | 2.38353e-18 | 2.21569e-20 |   3.24e-18    |
|            1 |        0.25 | 1.68562e-18 | 2.01396e-20 |   2.6449e-19  |
|            2 |        0.5  | 1.24297e-21 | 5.5185e-21  |   9e-20       |
|            3 |        0.75 | 6.6778e-22  | 2.54612e-21 |   4.48443e-20 |
|            4 |        1    | 6.80032e-22 | 3.33945e-21 |   2.67769e-20 |</div>



<br>

### Baseline Fit

We'll quickly plot the observed data alongside the smoothed estimate provided in the raw data

```python
fig, ax = plt.subplots(dpi=150)

ax.scatter(df_LIGO['frequency'], df_LIGO['H1'], color='k', linewidth=0, s=1)
ax.plot(df_LIGO['frequency'], df_LIGO['H1_smoothed'], color='r', alpha=1, label='Existing Smoothing')

ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(frameon=False)
eda.hide_spines(ax)
```


![png](./img/nbs/output_5_0.png)


<br>

### LOWESS Fit

```python
x = df_LIGO['frequency'].values
y = np.log(df_LIGO['H1']).values

df_bootstrap = lowess.bootstrap_model(x, y, num_runs=2500, frac=0.2, num_fits=30)

df_bootstrap.head()
```


<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:0; max-width:15ex; vertical-align:middle; text-align:right"></span>
<progress style="width:60ex" max="2500" value="2500" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">2500/2500</span>
<span class="Time-label">[05:54<00:00, 0.14s/it]</span></div>


    c:\users\ayrto\desktop\phd\analysis\merit-order-effect\moepy\lowess.py:145: FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.
    To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.
      betas = np.linalg.lstsq(A, b)[0]
    




|    x |        0 |        1 |        2 |        3 |        4 |        5 |        6 |        7 |        8 |        9 | ...   |     2490 |     2491 |     2492 |     2493 |     2494 |     2495 |     2496 |     2497 |     2498 |     2499 |
|-----:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|:------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| 0    | -52.6204 | -52.6989 | -52.6539 | -52.4245 | -52.7754 | -52.5348 | -52.5495 | -52.6225 | -52.4652 | -52.4438 | ...   | -52.5187 | -52.474  | -52.556  | -52.3936 | -52.4217 | -52.4993 | -52.6208 | -52.5166 | -52.5544 | -52.3874 |
| 0.25 | -52.6211 | -52.6994 | -52.6546 | -52.4255 | -52.7758 | -52.5356 | -52.5503 | -52.6232 | -52.4661 | -52.4448 | ...   | -52.5195 | -52.4749 | -52.5569 | -52.3946 | -52.4227 | -52.5002 | -52.6214 | -52.5173 | -52.5552 | -52.3884 |
| 0.5  | -52.6218 | -52.7    | -52.6553 | -52.4264 | -52.7763 | -52.5364 | -52.551  | -52.6239 | -52.467  | -52.4458 | ...   | -52.5204 | -52.4758 | -52.5577 | -52.3957 | -52.4237 | -52.5011 | -52.622  | -52.5181 | -52.556  | -52.3894 |
| 0.75 | -52.6225 | -52.7006 | -52.656  | -52.4274 | -52.7767 | -52.5372 | -52.5518 | -52.6246 | -52.4679 | -52.4467 | ...   | -52.5212 | -52.4767 | -52.5586 | -52.3967 | -52.4247 | -52.502  | -52.6226 | -52.5188 | -52.5568 | -52.3904 |
| 1    | -52.6232 | -52.7012 | -52.6566 | -52.4284 | -52.7771 | -52.538  | -52.5526 | -52.6253 | -52.4688 | -52.4477 | ...   | -52.522  | -52.4776 | -52.5594 | -52.3978 | -52.4257 | -52.5029 | -52.6233 | -52.5196 | -52.5577 | -52.3914 |</div>



<br>

Using `df_bootstrap` we can calculate the confidence interval of our predictions, the Pandas DataFrame `quantile` method makes this particularly simple.

```python
#exports
def get_confidence_interval(df_bootstrap, conf_pct=0.95):
    """Estimates the confidence interval of a prediction based on the bootstrapped estimates"""
    conf_margin = (1 - conf_pct)/2
    df_conf_intvl = pd.DataFrame(columns=['min', 'max'], index=df_bootstrap.index)
    
    df_conf_intvl['min'] = df_bootstrap.quantile(conf_margin, axis=1)
    df_conf_intvl['max'] = df_bootstrap.quantile(1-conf_margin, axis=1)
    
    return df_conf_intvl
```

```python
df_conf_intvl = get_confidence_interval(df_bootstrap, conf_pct=0.95)

# Plotting
fig, ax = plt.subplots(dpi=150)

ax.scatter(df_LIGO['frequency'], df_LIGO['H1'], color='k', linewidth=0, s=1, zorder=1)
ax.fill_between(df_conf_intvl.index, np.exp(df_conf_intvl['min']), np.exp(df_conf_intvl['max']), color='r', alpha=1, label='95% Confidence')

ax.set_yscale('log')
ax.legend(frameon=False)
eda.hide_spines(ax)
```


![png](./img/nbs/output_10_0.png)


<br>

We can see that we capture the middle and higher frequencies fairly well but due to the smaller number of data-points in the low frequency region the LOWESS fit is unable to model it as well. The `frac` could be decreased to improve this but then comes at the cost of processing more inaccurate estimates in the other regions. One way to address this could be to introduce an option where `frac` can be varied for each local regression model.

For now we'll just limit our prediction to the domain where the confidence interval is 'reasonable', we'll start by calculating the 95% confidence interval.

```python
s_95pct_conf_intvl = (df_bootstrap
                      .quantile([0.025, 0.975], axis=1)
                      .diff()
                      .dropna(how='all')
                      .T
                      .rename(columns={0.975: '95pct_pred_intvl'})
                      ['95pct_pred_intvl']
                     )

s_95pct_conf_intvl
```




    x
    0.00       0.378534
    0.25       0.377999
    0.50       0.377465
    0.75       0.376930
    1.00       0.376396
                 ...   
    2047.00    0.076900
    2047.25    0.076980
    2047.50    0.077059
    2047.75    0.077139
    2048.00    0.077222
    Name: 95pct_pred_intvl, Length: 8193, dtype: float64



<br>

We'll now define the 'reasonable' confidence interval threshold, in this case using the value that icludes 95% of the values.

```python
x_max_pct = 0.95

# Plotting
fig, ax = plt.subplots(dpi=150)

hist = sns.histplot(s_95pct_conf_intvl, ax=ax)

y_max = np.ceil(max([h.get_height() for h in hist.patches])/1e2)*1e2
x_max = s_95pct_conf_intvl.quantile(x_max_pct)
ax.plot([x_max, x_max], [0, y_max], linestyle='--', color='k', label='95% Coverage')

ax.set_ylim(0, y_max)
eda.hide_spines(ax)
```


![png](./img/nbs/output_14_0.png)


<br>

We'll then only plot our confidence interval when it's in this 'reasonable' range

```python
conf_intvl_idxs_to_keep = (s_95pct_conf_intvl<x_max).replace(False, np.nan).dropna().index

# Plotting
fig, ax = plt.subplots(dpi=150)

ax.scatter(df_LIGO['frequency'], df_LIGO['H1'], color='k', linewidth=0, s=1, zorder=1)
ax.fill_between(conf_intvl_idxs_to_keep, np.exp(df_conf_intvl.loc[conf_intvl_idxs_to_keep, 'min']), np.exp(df_conf_intvl.loc[conf_intvl_idxs_to_keep, 'max']), color='r', alpha=1, label='95% Confidence')

ax.set_yscale('log')
ax.legend(frameon=False)
eda.hide_spines(ax)
```


![png](./img/nbs/output_16_0.png)

