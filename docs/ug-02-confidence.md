# Bootstrapped LOWESS for Confidence Intervals



[![Binder](https://notebooks.gesis.org/binder/badge_logo.svg)](https://notebooks.gesis.org/binder/v2/gh/AyrtonB/Merit-Order-Effect/main?filepath=nbs%2Fug-02-confidence.ipynb)

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

df_bootstrap = lowess.bootstrap_model(x, y, num_runs=500, frac=0.2, num_fits=30)

df_bootstrap.head()
```


<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:0; max-width:15ex; vertical-align:middle; text-align:right"></span>
<progress style="width:60ex" max="500" value="500" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">500/500</span>
<span class="Time-label">[01:37<00:00, 0.19s/it]</span></div>





|    x |        0 |        1 |        2 |        3 |        4 |        5 |        6 |        7 |        8 |        9 | ...   |      490 |      491 |      492 |      493 |      494 |      495 |      496 |      497 |      498 |      499 |
|-----:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|:------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| 0    | -52.5227 | -52.724  | -52.4039 | -52.5183 | -52.5897 | -52.4715 | -52.5306 | -52.4727 | -52.5534 | -52.5212 | ...   | -52.5536 | -52.6798 | -52.508  | -52.554  | -52.4346 | -52.6022 | -52.355  | -52.5031 | -52.5719 | -52.5057 |
| 0.25 | -52.5235 | -52.7245 | -52.4049 | -52.5192 | -52.5904 | -52.4723 | -52.5314 | -52.4737 | -52.5542 | -52.522  | ...   | -52.5544 | -52.6804 | -52.5088 | -52.5547 | -52.4356 | -52.6029 | -52.3561 | -52.504  | -52.5727 | -52.5066 |
| 0.5  | -52.5244 | -52.725  | -52.4059 | -52.5201 | -52.5912 | -52.4732 | -52.5322 | -52.4746 | -52.555  | -52.5228 | ...   | -52.5553 | -52.681  | -52.5096 | -52.5555 | -52.4365 | -52.6037 | -52.3573 | -52.5048 | -52.5735 | -52.5075 |
| 0.75 | -52.5252 | -52.7256 | -52.4069 | -52.521  | -52.5919 | -52.4741 | -52.533  | -52.4755 | -52.5559 | -52.5236 | ...   | -52.5561 | -52.6816 | -52.5105 | -52.5562 | -52.4375 | -52.6044 | -52.3584 | -52.5057 | -52.5743 | -52.5084 |
| 1    | -52.5261 | -52.7261 | -52.4079 | -52.5218 | -52.5927 | -52.4749 | -52.5338 | -52.4764 | -52.5567 | -52.5244 | ...   | -52.5569 | -52.6822 | -52.5113 | -52.5569 | -52.4384 | -52.6051 | -52.3595 | -52.5065 | -52.5751 | -52.5093 |</div>



<br>

Using `df_bootstrap` we can calculate the confidence interval of our predictions, the Pandas DataFrame `quantile` method makes this particularly simple.

```python
df_conf_intvl = lowess.get_confidence_interval(df_bootstrap, conf_pct=0.95)

# Plotting
fig, ax = plt.subplots(dpi=150)

ax.scatter(df_LIGO['frequency'], df_LIGO['H1'], color='k', linewidth=0, s=1, zorder=1)
ax.fill_between(df_conf_intvl.index, np.exp(df_conf_intvl['min']), np.exp(df_conf_intvl['max']), color='r', alpha=1, label='95% Confidence')

ax.set_yscale('log')
ax.legend(frameon=False)
eda.hide_spines(ax)
```


![png](./img/nbs/output_9_0.png)


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
    0.00       0.404673
    0.25       0.404049
    0.50       0.403448
    0.75       0.402822
    1.00       0.402187
                 ...   
    2047.00    0.077741
    2047.25    0.077827
    2047.50    0.077913
    2047.75    0.077999
    2048.00    0.078085
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


![png](./img/nbs/output_13_0.png)


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


![png](./img/nbs/output_15_0.png)

