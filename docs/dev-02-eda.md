# Exploratory Data Analysis



[![Binder](https://notebooks.gesis.org/binder/badge_logo.svg)](https://notebooks.gesis.org/binder/v2/gh/AyrtonB/Merit-Order-Effect/main?filepath=nbs%2Fdev-02-eda.ipynb)

This notebook includes some visualisation and exploration of the price and fuel data for Germany and Great Britain

<br>

### Imports

```python
#exports
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtf
```

<br>

### Loading Data

```python
#exports
def load_EI_df(EI_fp):
    """Loads the electric insights data and returns a DataFrame"""
    df = pd.read_csv(EI_fp)

    df['local_datetime'] = pd.to_datetime(df['local_datetime'], utc=True)
    df = df.set_index('local_datetime')
    
    return df
```

```python
%%time

df = load_EI_df('../data/raw/electric_insights.csv')

df.head()
```

    Wall time: 7.95 s
    




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

We'll do the same for the German Energy-Charts and ENTSOE data

```python
#exports
def load_DE_df(EC_fp, ENTSOE_fp):
    """Loads the energy-charts and ENTSOE data and returns a DataFrame"""
    # Energy-Charts
    df_DE = pd.read_csv(EC_fp)

    df_DE['local_datetime'] = pd.to_datetime(df_DE['local_datetime'], utc=True)
    df_DE = df_DE.set_index('local_datetime')
    
    # ENTSOE
    df_ENTSOE = pd.read_csv(ENTSOE_fp)

    df_ENTSOE['local_datetime'] = pd.to_datetime(df_ENTSOE['local_datetime'], utc=True)
    df_ENTSOE = df_ENTSOE.set_index('local_datetime')
    
    # Combining data
    df_DE['demand'] = df_DE.sum(axis=1)
    
    s_price = df_ENTSOE['DE_price']
    df_DE['price'] = s_price[~s_price.index.duplicated(keep='first')]
    
    return df_DE
```

```python
df_DE = load_DE_df('../data/raw/energy_charts.csv', '../data/raw/ENTSOE_DE_price.csv')

df_DE.head()
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
      <th>Biomass</th>
      <th>Brown Coal</th>
      <th>Gas</th>
      <th>Hard Coal</th>
      <th>Hydro Power</th>
      <th>Oil</th>
      <th>Others</th>
      <th>Pumped Storage</th>
      <th>Seasonal Storage</th>
      <th>Solar</th>
      <th>Uranium</th>
      <th>Wind</th>
      <th>net_balance</th>
      <th>demand</th>
      <th>price</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-03 23:00:00+00:00</th>
      <td>3.637</td>
      <td>16.533</td>
      <td>4.726</td>
      <td>10.078</td>
      <td>2.331</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.052</td>
      <td>0.068</td>
      <td>0.0</td>
      <td>16.826</td>
      <td>0.635</td>
      <td>-1.229</td>
      <td>53.657</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2010-01-04 00:00:00+00:00</th>
      <td>3.637</td>
      <td>16.544</td>
      <td>4.856</td>
      <td>8.816</td>
      <td>2.293</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.038</td>
      <td>0.003</td>
      <td>0.0</td>
      <td>16.841</td>
      <td>0.528</td>
      <td>-1.593</td>
      <td>51.963</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2010-01-04 01:00:00+00:00</th>
      <td>3.637</td>
      <td>16.368</td>
      <td>5.275</td>
      <td>7.954</td>
      <td>2.299</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.032</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>16.846</td>
      <td>0.616</td>
      <td>-1.378</td>
      <td>51.649</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2010-01-04 02:00:00+00:00</th>
      <td>3.637</td>
      <td>15.837</td>
      <td>5.354</td>
      <td>7.681</td>
      <td>2.299</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.027</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>16.699</td>
      <td>0.630</td>
      <td>-1.624</td>
      <td>50.540</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2010-01-04 03:00:00+00:00</th>
      <td>3.637</td>
      <td>15.452</td>
      <td>5.918</td>
      <td>7.498</td>
      <td>2.301</td>
      <td>0.003</td>
      <td>0.0</td>
      <td>0.020</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>16.635</td>
      <td>0.713</td>
      <td>-0.731</td>
      <td>51.446</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



<br>

### Stacked-Fuels Time-Series

We'll create a stacked plot of the different generation types over time. We'll begin by cleaning the dataframe and merging columns so that it's ready for plotting, we'll also take the 7-day rolling average to make long-term trends clearer.

```python
#exports
def clean_df_for_plot(df, freq='7D'):
    """Cleans the electric insights dataframe for plotting"""
    fuel_order = ['Imports & Storage', 'nuclear', 'biomass', 'gas', 'coal', 'hydro', 'wind', 'solar']
    interconnectors = ['french', 'irish', 'dutch', 'belgian', 'ireland', 'northern_ireland']

    df = (df
          .copy()
          .assign(imports_storage=df[interconnectors+['pumped_storage']].sum(axis=1))
          .rename(columns={'imports_storage':'Imports & Storage'})
          .drop(columns=interconnectors+['demand', 'pumped_storage'])
          [fuel_order]
         )

    df_resampled = df.astype('float').resample(freq).mean()
    return df_resampled
```

```python
df_plot = clean_df_for_plot(df)

df_plot.head()
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
      <th>Imports &amp; Storage</th>
      <th>nuclear</th>
      <th>biomass</th>
      <th>gas</th>
      <th>coal</th>
      <th>hydro</th>
      <th>wind</th>
      <th>solar</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-01-01 00:00:00+00:00</th>
      <td>-0.039018</td>
      <td>5.768536</td>
      <td>0.0</td>
      <td>16.295098</td>
      <td>20.132420</td>
      <td>0.355890</td>
      <td>0.390015</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2009-01-08 00:00:00+00:00</th>
      <td>-0.921768</td>
      <td>5.582896</td>
      <td>0.0</td>
      <td>16.381083</td>
      <td>21.699726</td>
      <td>0.551753</td>
      <td>1.151545</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2009-01-15 00:00:00+00:00</th>
      <td>-0.024241</td>
      <td>5.559986</td>
      <td>0.0</td>
      <td>14.839983</td>
      <td>20.446309</td>
      <td>0.704382</td>
      <td>1.483002</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2009-01-22 00:00:00+00:00</th>
      <td>0.182830</td>
      <td>6.228411</td>
      <td>0.0</td>
      <td>14.467771</td>
      <td>20.590661</td>
      <td>0.562277</td>
      <td>0.938827</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2009-01-29 00:00:00+00:00</th>
      <td>0.120204</td>
      <td>6.799589</td>
      <td>0.0</td>
      <td>13.965650</td>
      <td>21.349710</td>
      <td>0.519632</td>
      <td>1.362611</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



<br>

We'll also define the colours we'll use for each fuel-type

N.b. the colour palette used is from [this paper](https://www.sciencedirect.com/science/article/pii/S0301421516307017)

```python
fuel_colour_dict_rgb = {
    'Imports & Storage' : (121,68,149), 
    'nuclear' : (77,157,87), 
    'biomass' : (168,125,81), 
    'gas' : (254,156,66), 
    'coal' : (122,122,122), 
    'hydro' : (50,120,196), 
    'wind' : (72,194,227), 
    'solar' : (255,219,65),
}
```

<br>

However we need to convert from rgb to matplotlib plotting colours (0-1 not 0-255)

```python
#exports
def rgb_2_plt_tuple(rgb_tuple):
    """converts a standard rgb set from a 0-255 range to 0-1"""
    plt_tuple = tuple([x/255 for x in rgb_tuple])
    return plt_tuple

def convert_fuel_colour_dict_to_plt_tuple(fuel_colour_dict_rgb):
    """Converts a dictionary of fuel colours to matplotlib colour values"""
    fuel_colour_dict_plt = fuel_colour_dict_rgb.copy()
    
    fuel_colour_dict_plt = {
        fuel: rgb_2_plt_tuple(rgb_tuple) 
        for fuel, rgb_tuple 
        in fuel_colour_dict_plt.items()
    }
    
    return fuel_colour_dict_plt
```

```python
fuel_colour_dict_plt = convert_fuel_colour_dict_to_plt_tuple(fuel_colour_dict_rgb)

sns.palplot(fuel_colour_dict_plt.values())
```


![png](./img/nbs/output_15_0.png)


<br>

Finally we can plot the stacked fuel plot itself

```python
#exports
def hide_spines(ax, positions=["top", "right"]):
    """
    Pass a matplotlib axis and list of positions with spines to be removed
    
    Parameters:
        ax:          Matplotlib axis object
        positions:   Python list e.g. ['top', 'bottom']
    """
    assert isinstance(positions, list), "Position must be passed as a list "

    for position in positions:
        ax.spines[position].set_visible(False)
        
def stacked_fuel_plot(df, fuel_colour_dict, ax=None, save_path=None, dpi=150):
    """Plots the electric insights fuel data as a stacked area graph"""
    df = df[fuel_colour_dict.keys()]
    
    if ax == None:
        fig = plt.figure(figsize=(10, 5), dpi=dpi)
        ax = plt.subplot()
    
    ax.stackplot(df.index.values, df.values.T, labels=df.columns.str.capitalize(), linewidth=0.25, edgecolor='white', colors=list(fuel_colour_dict.values()))

    plt.rcParams['axes.ymargin'] = 0
    ax.spines['bottom'].set_position('zero')
    hide_spines(ax)

    ax.set_xlim(df.index.min(), df.index.max())
    ax.legend(ncol=4, bbox_to_anchor=(0.85, 1.15), frameon=False)
    ax.set_ylabel('Generation (GW)')

    if save_path:
        fig.savefig(save_path)
        
    return ax
```

```python
stacked_fuel_plot(df_plot, fuel_colour_dict_plt, dpi=250)
```




    <AxesSubplot:ylabel='Generation (GW)'>




![png](./img/nbs/output_18_1.png)

