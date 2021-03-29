# Tables & Figures Generation



[![Binder](https://notebooks.gesis.org/binder/badge_logo.svg)](https://notebooks.gesis.org/binder/v2/gh/AyrtonB/Merit-Order-Effect/main?filepath=nbs%2Fdev-09-tables-and-figures.ipynb)

This notebook provides a programmatic workflow for generating the tables used in the MOE paper, as well as the diagram to show the time-adaptive smoothing weights.

<br>

### Imports

```python
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import Latex, JSON

from moepy import eda, lowess
```

<br>

### Tables

##### Power Systems Overview

We'll first load in the DE data

```python
df_DE = eda.load_DE_df('../data/raw/energy_charts.csv', '../data/raw/ENTSOE_DE_price.csv')

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

Clean it up then calculate the relevant summary statistics

```python
s_DE_RES_output = df_DE[['Wind', 'Solar']].sum(axis=1)
s_DE_demand = df_DE['demand']
s_DE_price = df_DE['price']

s_DE_RES_pct = s_DE_RES_output/s_DE_demand

DE_2020_RES_pct = s_DE_RES_pct['2020'].mean()
DE_2020_demand_avg = s_DE_demand['2020'].mean()
DE_2020_price_avg = s_DE_price['2020'].mean()

DE_2020_RES_pct, DE_2020_demand_avg, DE_2020_price_avg
```




    (0.3593124152992342, 55.956133452868855, 30.469415917112606)



<br>

We'll also estimate the carbon intensity

```python
DE_fuel_to_co2_intensity = {
    'Biomass': 0.39, 
    'Brown Coal': 0.36, 
    'Gas': 0.23, 
    'Hard Coal': 0.34, 
    'Hydro Power': 0, 
    'Oil': 0.28,
    'Others': 0, 
    'Pumped Storage': 0, 
    'Seasonal Storage': 0, 
    'Solar': 0, 
    'Uranium': 0,
    'Wind': 0, 
    'net_balance': 0 
}

s_DE_emissions_tonnes = (df_DE
                         [DE_fuel_to_co2_intensity.keys()]
                         .multiply(1e3) # converting to MWh
                         .multiply(DE_fuel_to_co2_intensity.values())
                         .sum(axis=1)
                        )

s_DE_emissions_tonnes = s_DE_emissions_tonnes[s_DE_emissions_tonnes>2000]
s_DE_carbon_intensity = s_DE_emissions_tonnes/s_DE_demand.loc[s_DE_emissions_tonnes.index]

DE_2020_emissions_tonnes = s_DE_emissions_tonnes['2020'].mean()
DE_2020_ci_avg = s_DE_carbon_intensity['2020'].mean()

DE_2020_emissions_tonnes, DE_2020_ci_avg
```




    (8448.292069623136, 153.80385402105972)



<br>

We'll do the same for GB

```python
# Loading in
df_EI = pd.read_csv('../data/raw/electric_insights.csv')

df_EI = df_EI.set_index('local_datetime')
df_EI.index = pd.to_datetime(df_EI.index, utc=True)

# Extracting RES, demand, and price series
s_GB_RES = df_EI[['wind', 'solar']].sum(axis=1)
s_GB_demand = df_EI['demand']
s_GB_price = df_EI['day_ahead_price']

# Generating carbon intensity series
GB_fuel_to_co2_intensity = {
    'nuclear': 0, 
    'biomass': 0.121, # from EI 
    'coal': 0.921, # DUKES 2018 value
    'gas': 0.377, # DUKES 2018 value (lower than many CCGT estimates, let alone OCGT)
    'hydro': 0, 
    'pumped_storage': 0, 
    'solar': 0,
    'wind': 0,
    'belgian': 0.4,  
    'dutch': 0.474, # from EI 
    'french': 0.053, # from EI 
    'ireland': 0.458, # from EI 
    'northern_ireland': 0.458 # from EI 
}

s_GB_emissions_tonnes = (df_EI
                         [GB_fuel_to_co2_intensity.keys()]
                         .multiply(1e3*0.5) # converting to MWh
                         .multiply(GB_fuel_to_co2_intensity.values())
                         .sum(axis=1)
                        )

s_GB_emissions_tonnes = s_GB_emissions_tonnes[s_GB_emissions_tonnes>2000]
s_GB_carbon_intensity = s_GB_emissions_tonnes/s_GB_demand.loc[s_GB_emissions_tonnes.index]

# Calculating 2020 averages
GB_2020_emissions_tonnes = s_GB_emissions_tonnes['2020'].mean()
GB_2020_ci_avg = s_GB_carbon_intensity['2020'].mean()
GB_2020_RES_pct = (s_GB_RES['2020']/s_GB_demand['2020']).mean()
GB_2020_demand_avg = s_GB_demand['2020'].mean()
GB_2020_price_avg = s_GB_price['2020'].mean()
```

<br>

Then combine the results in a single table

```python
system_overview_data = {
    'Germany': {
        'Average Solar/Wind Generation (%)': round(100*DE_2020_RES_pct, 2),
        'Average Demand (GW)': round(DE_2020_demand_avg, 2),
        'Average Price ([EUR,GBP]/MWh)': round(DE_2020_price_avg, 2),
        'Average Carbon Intensity  (gCO2/kWh)': round(DE_2020_ci_avg, 2),
    },
    'Great Britain': {
        'Average Solar/Wind Generation (%)': round(100*GB_2020_RES_pct, 2),
        'Average Demand (GW)': round(GB_2020_demand_avg, 2),
        'Average Price ([EUR,GBP]/MWh)': round(GB_2020_price_avg, 2),
        'Average Carbon Intensity  (gCO2/kWh)': round(GB_2020_ci_avg, 2),
    }
}

df_system_overview = pd.DataFrame(system_overview_data).T

df_system_overview.head()
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
      <th>Average Solar/Wind Generation (%)</th>
      <th>Average Demand (GW)</th>
      <th>Average Price ([EUR,GBP]/MWh)</th>
      <th>Average Carbon Intensity  (gCO2/kWh)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Germany</th>
      <td>35.93</td>
      <td>55.96</td>
      <td>30.47</td>
      <td>153.80</td>
    </tr>
    <tr>
      <th>Great Britain</th>
      <td>29.83</td>
      <td>30.61</td>
      <td>33.77</td>
      <td>101.17</td>
    </tr>
  </tbody>
</table>
</div>



<br>

Which we'll then output as a LaTeX table

```python
get_lined_column_format = lambda n_cols:''.join(n_cols*['|l']) + '|'

caption = 'Systems overview for 2020'
label = 'overview_table'
column_format = get_lined_column_format(df_system_overview.shape[1]+1)

latex_str = df_system_overview.to_latex(column_format=column_format, caption=caption, label=label)

latex_replacements = {
    'CO2': 'CO\\textsubscript{2}',
    '\\\\\n': '\\\\ \\midrule\n',
    'midrule': 'hline',
    'toprule': 'hline',
    'bottomrule': '',
    '\n\\\n': '\n',
    '\\hline\n\\hline': '\\hline'
}

for old, new in latex_replacements.items():
    latex_str = latex_str.replace(old, new)

Latex(latex_str)
```




\begin{table}
\centering
\caption{Systems overview for 2020}
\label{overview_table}
\begin{tabular}{|l|l|l|l|l|}
\hline
{} &  Average Solar/Wind Generation (\%) &  Average Demand (GW) &  Average Price ([EUR,GBP]/MWh) &  Average Carbon Intensity  (gCO\textsubscript{2}/kWh) \\ \hline
Germany       &                              35.93 &                55.96 &                          30.47 &                                153.80 \\ \hline
Great Britain &                              29.83 &                30.61 &                          33.77 &                                101.17 \\ \hline
\end{tabular}
\end{table}




<br>

##### Carbon Intensity Estimates

We'll clean up our GB carbon intensity estimates

```python
def clean_idxs(s):
    s.index = s.index.str.replace('_', ' ').str.title()
    return s

df_GB_non0_co2_intensity = (pd
                            .Series(GB_fuel_to_co2_intensity)
                            .replace(0, np.nan)
                            .dropna()
                            .drop(['belgian', 'northern_ireland'])
                            .pipe(clean_idxs)
                            .multiply(1e3)
                            .astype(int)
                            .to_frame()
                            .T
                            .rename({0: 'gCO2/kWh'})
                           )

df_GB_non0_co2_intensity
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
      <th>Coal</th>
      <th>Gas</th>
      <th>Dutch</th>
      <th>French</th>
      <th>Ireland</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gCO2/kWh</th>
      <td>121</td>
      <td>921</td>
      <td>377</td>
      <td>474</td>
      <td>53</td>
      <td>458</td>
    </tr>
  </tbody>
</table>
</div>



<br>

And output them as a LaTeX table

```python
caption = 'Carbon intensity factors for fuel-types and interconnection on the GB power system'
label = 'GB_co2_intensity_table'
column_format = get_lined_column_format(df_GB_non0_co2_intensity.shape[1]+1)

latex_str = df_GB_non0_co2_intensity.to_latex(column_format=column_format, caption=caption, label=label)

latex_replacements = {
    'CO2': 'CO\\textsubscript{2}',
    '\\\\\n': '\\\\ \\midrule\n',
    'midrule': 'hline',
    'toprule': 'hline',
    'bottomrule': '',
    '\n\\\n': '\n',
    '\\hline\n\\hline': '\\hline'
}

for old, new in latex_replacements.items():
    latex_str = latex_str.replace(old, new)

Latex(latex_str)
```




\begin{table}
\centering
\caption{Carbon intensity factors for fuel-types and interconnection on the GB power system}
\label{GB_co2_intensity_table}
\begin{tabular}{|l|l|l|l|l|l|l|}
\hline
{} &  Biomass &  Coal &  Gas &  Dutch &  French &  Ireland \\ \hline
gCO\textsubscript{2}/kWh &      121 &   921 &  377 &    474 &      53 &      458 \\ \hline
\end{tabular}
\end{table}




<br>

We'll then do the same for DE

```python
df_DE_non0_co2_intensity = (pd
                            .Series(DE_fuel_to_co2_intensity)
                            .replace(0, np.nan)
                            .dropna()
                            [['Biomass', 'Brown Coal', 'Hard Coal', 'Gas', 'Oil']]
                            .pipe(clean_idxs)
                            .multiply(1e3)
                            .astype(int)
                            .to_frame()
                            .T
                            .rename({0: 'gCO2/kWh'})
                           )

df_DE_non0_co2_intensity
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
      <th>Hard Coal</th>
      <th>Gas</th>
      <th>Oil</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gCO2/kWh</th>
      <td>390</td>
      <td>360</td>
      <td>340</td>
      <td>230</td>
      <td>280</td>
    </tr>
  </tbody>
</table>
</div>



```python
caption = 'Carbon intensity factors for fuel-types and interconnection on the DE power system'
label = 'DE_co2_intensity_table'
column_format = get_lined_column_format(df_DE_non0_co2_intensity.shape[1]+1)

latex_str = df_DE_non0_co2_intensity.to_latex(column_format=column_format, caption=caption, label=label)

for old, new in latex_replacements.items():
    latex_str = latex_str.replace(old, new)

Latex(latex_str)
```




\begin{table}
\centering
\caption{Carbon intensity factors for fuel-types and interconnection on the DE power system}
\label{DE_co2_intensity_table}
\begin{tabular}{|l|l|l|l|l|l|}
\hline
{} &  Biomass &  Brown Coal &  Hard Coal &  Gas &  Oil \\ \hline
gCO\textsubscript{2}/kWh &      390 &         360 &        340 &  230 &  280 \\ \hline
\end{tabular}
\end{table}




<br>

##### Electricity Price Forecasting Metrics

We'll start by loading in our previously saved model metrics

```python
with open('../data/results/price_model_accuracy_metrics.json', 'r') as fp:
    model_accuracy_metrics = json.load(fp)
    
JSON(model_accuracy_metrics)
```




    <IPython.core.display.JSON object>



<br>

We'll parse the MAE results into a new table

```python
model_accuracy_data = {
    'Germany': {
        'Dispatchable Load': round(model_accuracy_metrics['DE_dispatch']['mean_abs_err'], 2),
        'Total Load': round(model_accuracy_metrics['DE_demand']['mean_abs_err'], 2),
    },
    'Great Britain': {
        'Dispatchable Load': round(model_accuracy_metrics['GB_dispatch']['mean_abs_err'], 2),
        'Total Load': round(model_accuracy_metrics['GB_demand']['mean_abs_err'], 2),
    }
}

df_model_accuracy = pd.DataFrame(model_accuracy_data).T

df_model_accuracy.head()
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
      <th>Dispatchable Load</th>
      <th>Total Load</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Germany</th>
      <td>5.85</td>
      <td>18.28</td>
    </tr>
    <tr>
      <th>Great Britain</th>
      <td>6.56</td>
      <td>8.42</td>
    </tr>
  </tbody>
</table>
</div>



<br>

Which we'll output as a LaTeX table

```python
caption = 'Price forecasting model accuracy when regressing against dispatchable and total load for GB and DE.'
label = 'model_accuracy_table'
column_format = get_lined_column_format(df_model_accuracy.shape[1]+1)

latex_str = df_model_accuracy.to_latex(column_format=column_format, caption=caption, label=label)

for old, new in latex_replacements.items():
    latex_str = latex_str.replace(old, new)

Latex(latex_str)
```




\begin{table}
\centering
\caption{Price forecasting model accuracy when regressing against dispatchable and total load for GB and DE.}
\label{model_accuracy_table}
\begin{tabular}{|l|l|l|}
\hline
{} &  Dispatchable Load &  Total Load \\ \hline
Germany       &               5.85 &       18.28 \\ \hline
Great Britain &               6.56 &        8.42 \\ \hline
\end{tabular}
\end{table}




<br>

##### Price and CO2 MOE Results

We'll first load in all of the price and carbon MOE time-series

```python
def set_dt_idx(df, dt_idx_col='local_datetime'):
    df = df.set_index(dt_idx_col)
    df.index = pd.to_datetime(df.index, utc=True)
    
    return df

df_GB_price_results_ts = pd.read_csv('../data/results/GB_price.csv').pipe(set_dt_idx)
df_DE_price_results_ts = pd.read_csv('../data/results/DE_price.csv').pipe(set_dt_idx)
df_GB_carbon_results_ts = pd.read_csv('../data/results/GB_carbon.csv').pipe(set_dt_idx)
df_DE_carbon_results_ts = pd.read_csv('../data/results/DE_carbon.csv').pipe(set_dt_idx)

df_GB_price_results_ts.head()
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
      <th>prediction</th>
      <th>counterfactual</th>
      <th>observed</th>
      <th>moe</th>
    </tr>
    <tr>
      <th>local_datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-01-01 00:00:00+00:00</th>
      <td>37.203441</td>
      <td>37.313379</td>
      <td>58.05</td>
      <td>0.109938</td>
    </tr>
    <tr>
      <th>2009-01-01 00:30:00+00:00</th>
      <td>37.313379</td>
      <td>37.535135</td>
      <td>56.33</td>
      <td>0.221756</td>
    </tr>
    <tr>
      <th>2009-01-01 01:00:00+00:00</th>
      <td>36.768513</td>
      <td>36.985087</td>
      <td>52.98</td>
      <td>0.216574</td>
    </tr>
    <tr>
      <th>2009-01-01 01:30:00+00:00</th>
      <td>35.595162</td>
      <td>35.807631</td>
      <td>50.39</td>
      <td>0.212469</td>
    </tr>
    <tr>
      <th>2009-01-01 02:00:00+00:00</th>
      <td>34.849422</td>
      <td>35.063119</td>
      <td>48.70</td>
      <td>0.213697</td>
    </tr>
  </tbody>
</table>
</div>



<br>

We'll then calculate their summary statistics

```python
MOE_results_data = {
    'Germany': {
        'Price ([EUR,GBP]/MWh)': round(df_DE_price_results_ts.loc['2020', 'moe'].mean(), 2),
        'Price (%)': round(100*(df_DE_price_results_ts.loc['2020', 'moe']*df_DE['demand']).sum()/((df_DE_price_results_ts.loc['2020', 'observed']+df_DE_price_results_ts.loc['2020', 'moe'])*df_DE['demand']).sum(), 2),
        'Carbon (Tonnes/h)': round(df_DE_carbon_results_ts.loc['2020', 'moe'].mean(), 2),
        'Carbon (%)': round(100*(df_DE_carbon_results_ts.loc['2020', 'moe'].sum()/(df_DE_carbon_results_ts.loc['2020', 'observed']+df_DE_carbon_results_ts.loc['2020', 'moe']).sum()).mean(), 2)
    },
    'Great Britain': {
        'Price ([EUR,GBP]/MWh)': round(df_GB_price_results_ts.loc['2020', 'moe'].mean(), 2),
        'Price (%)': round(100*(df_GB_price_results_ts.loc['2020', 'moe']*df_EI['demand']).sum()/((df_GB_price_results_ts.loc['2020', 'observed']+df_GB_price_results_ts.loc['2020', 'moe'])*df_EI['demand']).sum(), 2),
        'Carbon (Tonnes/h)': round(df_GB_carbon_results_ts.loc['2020', 'moe'].mean(), 2), # doubled to make it the same hourly rate as DE
        'Carbon (%)': round(100*(df_GB_carbon_results_ts.loc['2020', 'moe'].sum()/(df_GB_carbon_results_ts.loc['2020', 'observed']+df_GB_carbon_results_ts.loc['2020', 'moe']).sum()).mean(), 2)
    }
}

df_MOE_results = (pd
                  .DataFrame(MOE_results_data)
                 )

df_MOE_results.head()
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
      <th>Germany</th>
      <th>Great Britain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Price ([EUR,GBP]/MWh)</th>
      <td>22.17</td>
      <td>13.89</td>
    </tr>
    <tr>
      <th>Price (%)</th>
      <td>43.43</td>
      <td>29.66</td>
    </tr>
    <tr>
      <th>Carbon (Tonnes/h)</th>
      <td>5563.22</td>
      <td>1657.88</td>
    </tr>
    <tr>
      <th>Carbon (%)</th>
      <td>39.70</td>
      <td>37.89</td>
    </tr>
  </tbody>
</table>
</div>



<br>

And export the output as a LaTeX table

```python
caption = '2020 Merit Order Effect results overview (weighted by volume).'
label = 'moe_results_table'
column_format = get_lined_column_format(df_MOE_results.shape[1]+1)

latex_str = df_MOE_results.to_latex(column_format=column_format, caption=caption, label=label)

for old, new in latex_replacements.items():
    latex_str = latex_str.replace(old, new)

Latex(latex_str)
```




\begin{table}
\centering
\caption{2020 Merit Order Effect results overview (weighted by volume).}
\label{moe_results_table}
\begin{tabular}{|l|l|l|}
\hline
{} &  Germany &  Great Britain \\ \hline
Price ([EUR,GBP]/MWh) &    22.17 &          13.89 \\ \hline
Price (\%)             &    43.43 &          29.66 \\ \hline
Carbon (Tonnes/h)     &  5563.22 &        1657.88 \\ \hline
Carbon (\%)            &    39.70 &          37.89 \\ \hline
\end{tabular}
\end{table}




<br>

##### Literature Review

Lastly we'll create our largest table, containing results from across the literature

```python
lit_results_data = [
    {
        'Study': 'Sensfuss et al. (2008)',
        'MOE': '7.83 €/MWh',
        'Period': '2006',
        'Region': 'Germany',
        'Method': 'ESS',
    },
    {
        'Study': 'Weigt (2009)',
        'MOE': '10 €/MWh',
        'Period': '2006-2008',
        'Region': 'Germany',
        'Method': 'ESS',
    },
    {
        'Study': 'Keles et al. (2013)',
        'MOE': '5.90 €/MWh',
        'Period': '2006–2009',
        'Region': 'Germany',
        'Method': 'RPR',
    },
    {
        'Study': 'Mulder and Scholtens (2013)',
        'MOE': '0.03% price decrease per p.p increase in wind speeds',
        'Period': '2006–2011',
        'Region': 'Germany',
        'Method': 'RPR',
    },
    {
        'Study': 'Tveten et al. (2013)',
        'MOE': '5.25 €/MWh (solar)',
        'Period': '2006-2011',
        'Region': 'Germany',
        'Method': 'RPR',
    },
    {
        'Study': 'Wurzburg et al. (2013)',
        'MOE': '2% price decrease',
        'Period': '2010-2012',
        'Region': 'Germany & Austria',
        'Method': 'RPR',
    },
    {
        'Study': 'Cludius et al. (2014)',
        'MOE': '8 €/MWh',
        'Period': '2010-2012',
        'Region': 'Germany',
        'Method': 'RPR',
    },
    {
        'Study': 'Ketterer (2014)',
        'MOE': '0.1-1.46% price decrease per p.p increase in wind generation',
        'Period': '2006-2012',
        'Region': 'Germany',
        'Method': 'RPR',
    },
    {
        'Study': 'Ederer (2015)',
        'MOE': '1.3% price decrease per annual TWh of wind',
        'Period': '2006-2014',
        'Region': 'Germany',
        'Method': 'MSS',
    },
    {
        'Study': 'Kyritsis et al. (2017)',
        'MOE': '-',
        'Period': '2010-2015',
        'Region': 'Germany',
        'Method': 'RPR',
    },
    {
        'Study': 'Bublitz et al. (2017)',
        'MOE': '5.40 €/MWh',
        'Period': '2011-2015',
        'Region': 'Germany',
        'Method': 'ESS',
    },
    {
        'Study': 'Bublitz et al. (2017)',
        'MOE': '6.80 €/MWh',
        'Period': '2011-2015',
        'Region': 'Germany',
        'Method': 'RPR',
    },
    {
        'Study': 'de Miera et al. (2008)',
        'MOE': '8.6-25.1% price decrease',
        'Period': '2005-2007',
        'Region': 'Spain',
        'Method': 'ESS',
    },
    {
        'Study': 'Gelabert et al. (2011)',
        'MOE': '3.7% price decrease',
        'Period': '2005-2012',
        'Region': 'Spain',
        'Method': 'RPR',
    },
    {
        'Study': 'Ciarreta et al. (2014)',
        'MOE': '25-45 €/MWh',
        'Period': '2008–2012',
        'Region': 'Spain',
        'Method': 'ESS',
    },
    {
        'Study': 'Clo et al. (2015)',
        'MOE': '2.3 €/MWh (solar), 4.2 €/MWh (wind)',
        'Period': '2005–2013',
        'Region': 'Italy',
        'Method': 'RPR',
    },
    {
        'Study': 'Munksgaard and Morthorst (2008)',
        'MOE': '1-4 €/MWh',
        'Period': '2004-2006',
        'Region': 'Denmark',
        'Method': 'RPR',
    },
    {
        'Study': 'Jonsson et al. (2010)',
        'MOE': '-',
        'Period': '2006-2007',
        'Region': 'Denmark',
        'Method': 'RPR',
    },
    {
        'Study': 'Denny et al. (2017)',
        'MOE': '3.40 €/MWh per GWh (wind)',
        'Period': '2009',
        'Region': 'Ireland',
        'Method': 'RPR',
    },
    {
        'Study': 'Lunackova et al. (2017)',
        'MOE': '1.2% price decrease per 10% increase in RES',
        'Period': '2010-2015',
        'Region': 'Czech Republic',
        'Method': 'RPR',
    },
    {
        'Study': 'Dillig et al. (2016)',
        'MOE': '50.29 €/MWh',
        'Period': '2011-2013',
        'Region': 'Germany',
        'Method': 'MSS',
    },
    {
        'Study': 'McConnell et al. (2013)',
        'MOE': '8.6% price decrease',
        'Period': '2009-2010',
        'Region': 'Australia',
        'Method': 'MSS',
    },
    {
        'Study': 'Moreno et al. (2012)',
        'MOE': '0.018% price increase per p.p. increase in RES penetration',
        'Period': '1998–2009',
        'Region': 'EU-27',
        'Method': 'RPR',
    },
    {
        'Study': 'Woo et al. (2011)',
        'MOE': '0.32-1.53 $/MWh',
        'Period': '2007-2010',
        'Region': 'Texas',
        'Method': 'RPR',
    },
    {
        'Study': 'Kaufmann and Vaid (2016)',
        'MOE': '0.26-1.86 $/MWh (solar)',
        'Period': '2010-2012',
        'Region': 'Massachusetts',
        'Method': 'RPR',
    },
    {
        'Study': 'Woo et al. (2016)',
        'MOE': '5.3 \$/MWh (solar) and 3.3 \$/MWh (wind) per GWh of RES',
        'Period': '2012-2015',
        'Region': 'California',
        'Method': 'RPR',
    },
    {
        'Study': 'Paraschiv et al. (2014)',
        'MOE': '0.15% price decrease per MWh of RES',
        'Period': '2010-2013',
        'Region': 'Germany',
        'Method': 'RPR',
    },
    {
        'Study': 'O\'Mahoney and Denny (2011)',
        'MOE': '12% price decrease',
        'Period': '2009',
        'Region': 'Ireland',
        'Method': 'RPR',
    },
    {
        'Study': 'Hildmann et al. (2015)',
        'MOE': '13.4-18.6 €/MWh',
        'Period': '2011-2013',
        'Region': 'Germany and Austria',
        'Method': 'MSS',
    },
    {
        'Study': 'Gil et al. (2012)',
        'MOE': '9.72 €/MWh',
        'Period': '2007-2010',
        'Region': 'Spain',
        'Method': 'RPR',
    },
#     { # Removed due to language barrier preventing method from being discerned
#         'Study': 'Weber and Woll (2007)',
#         'MOE': '4 €/MWh',
#         'Period': '2006',
#         'Region': 'Germany',
#         'Method': '-',
#     },
    {
        'Study': 'Halttunen et al. (2021)',
        'MOE': '0.631 €/MWh per p.p. increase in RES penetration',
        'Period': '2012-2019',
        'Region': 'Germany',
        'Method': 'RPR',
    },
    {
        'Study': 'Halttunen et al. (2021)',
        'MOE': '0.482 €/MWh per p.p. increase in RES penetration',
        'Period': '2010-2019',
        'Region': 'Germany',
        'Method': 'RPR',
    }
]

df_lit_results = pd.DataFrame(lit_results_data)

df_lit_results['Study Year'] = df_lit_results['Study'].str.split('(').str[1].str.replace(')', '').astype(int)
df_lit_results = df_lit_results.sort_values(['Method', 'Study Year', 'Study']).drop(columns=['Study Year']).reset_index(drop=True)

df_lit_results.head()
```

    <ipython-input-20-15bc63a4c27e>:237: FutureWarning: The default value of regex will change from True to False in a future version. In addition, single character regular expressions will*not* be treated as literal strings when regex=True.
      df_lit_results['Study Year'] = df_lit_results['Study'].str.split('(').str[1].str.replace(')', '').astype(int)
    




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
      <th>Study</th>
      <th>MOE</th>
      <th>Period</th>
      <th>Region</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sensfuss et al. (2008)</td>
      <td>7.83 €/MWh</td>
      <td>2006</td>
      <td>Germany</td>
      <td>ESS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>de Miera et al. (2008)</td>
      <td>8.6-25.1% price decrease</td>
      <td>2005-2007</td>
      <td>Spain</td>
      <td>ESS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Weigt (2009)</td>
      <td>10 €/MWh</td>
      <td>2006-2008</td>
      <td>Germany</td>
      <td>ESS</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ciarreta et al. (2014)</td>
      <td>25-45 €/MWh</td>
      <td>2008–2012</td>
      <td>Spain</td>
      <td>ESS</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bublitz et al. (2017)</td>
      <td>5.40 €/MWh</td>
      <td>2011-2015</td>
      <td>Germany</td>
      <td>ESS</td>
    </tr>
  </tbody>
</table>
</div>



<br>

We'll also export this as a LaTeX table

```python
caption = 'Results overview from the MOE literature'
label = 'lit_results_table'
column_format = get_lined_column_format(df_lit_results.shape[1]+1)

latex_str = df_lit_results.to_latex(column_format=column_format, caption=caption, label=label, index=False)

for old, new in latex_replacements.items():
    latex_str = latex_str.replace(old, new)

Latex(latex_str)
```




\begin{table}
\centering
\caption{Results overview from the MOE literature}
\label{lit_results_table}
\begin{tabular}{|l|l|l|l|l|l|}
\hline
                          Study &                                                MOE &    Period &              Region & Method \\ \hline
         Sensfuss et al. (2008) &                                         7.83 €/MWh &      2006 &             Germany &    ESS \\ \hline
         de Miera et al. (2008) &                           8.6-25.1\% price decrease & 2005-2007 &               Spain &    ESS \\ \hline
                   Weigt (2009) &                                           10 €/MWh & 2006-2008 &             Germany &    ESS \\ \hline
         Ciarreta et al. (2014) &                                        25-45 €/MWh & 2008–2012 &               Spain &    ESS \\ \hline
          Bublitz et al. (2017) &                                         5.40 €/MWh & 2011-2015 &             Germany &    ESS \\ \hline
        McConnell et al. (2013) &                                8.6\% price decrease & 2009-2010 &           Australia &    MSS \\ \hline
                  Ederer (2015) &         1.3\% price decrease per annual TWh of wind & 2006-2014 &             Germany &    MSS \\ \hline
         Hildmann et al. (2015) &                                    13.4-18.6 €/MWh & 2011-2013 & Germany and Austria &    MSS \\ \hline
           Dillig et al. (2016) &                                        50.29 €/MWh & 2011-2013 &             Germany &    MSS \\ \hline
Munksgaard and Morthorst (2008) &                                          1-4 €/MWh & 2004-2006 &             Denmark &    RPR \\ \hline
          Jonsson et al. (2010) &                                                  - & 2006-2007 &             Denmark &    RPR \\ \hline
         Gelabert et al. (2011) &                                3.7\% price decrease & 2005-2012 &               Spain &    RPR \\ \hline
     O'Mahoney and Denny (2011) &                                 12\% price decrease &      2009 &             Ireland &    RPR \\ \hline
              Woo et al. (2011) &                                    0.32-1.53 \$/MWh & 2007-2010 &               Texas &    RPR \\ \hline
              Gil et al. (2012) &                                         9.72 €/MWh & 2007-2010 &               Spain &    RPR \\ \hline
           Moreno et al. (2012) & 0.018\% price increase per p.p. increase in RES ... & 1998–2009 &               EU-27 &    RPR \\ \hline
            Keles et al. (2013) &                                         5.90 €/MWh & 2006–2009 &             Germany &    RPR \\ \hline
    Mulder and Scholtens (2013) & 0.03\% price decrease per p.p increase in wind s... & 2006–2011 &             Germany &    RPR \\ \hline
           Tveten et al. (2013) &                                 5.25 €/MWh (solar) & 2006-2011 &             Germany &    RPR \\ \hline
         Wurzburg et al. (2013) &                                  2\% price decrease & 2010-2012 &   Germany \& Austria &    RPR \\ \hline
          Cludius et al. (2014) &                                            8 €/MWh & 2010-2012 &             Germany &    RPR \\ \hline
                Ketterer (2014) & 0.1-1.46\% price decrease per p.p increase in wi... & 2006-2012 &             Germany &    RPR \\ \hline
        Paraschiv et al. (2014) &                0.15\% price decrease per MWh of RES & 2010-2013 &             Germany &    RPR \\ \hline
              Clo et al. (2015) &                2.3 €/MWh (solar), 4.2 €/MWh (wind) & 2005–2013 &               Italy &    RPR \\ \hline
       Kaufmann and Vaid (2016) &                            0.26-1.86 \$/MWh (solar) & 2010-2012 &       Massachusetts &    RPR \\ \hline
              Woo et al. (2016) & 5.3 \textbackslash \$/MWh (solar) and 3.3 \textbackslash \$/MWh (wind) per GW... & 2012-2015 &          California &    RPR \\ \hline
          Bublitz et al. (2017) &                                         6.80 €/MWh & 2011-2015 &             Germany &    RPR \\ \hline
            Denny et al. (2017) &                          3.40 €/MWh per GWh (wind) &      2009 &             Ireland &    RPR \\ \hline
         Kyritsis et al. (2017) &                                                  - & 2010-2015 &             Germany &    RPR \\ \hline
        Lunackova et al. (2017) &        1.2\% price decrease per 10\% increase in RES & 2010-2015 &      Czech Republic &    RPR \\ \hline
        Halttunen et al. (2021) &   0.631 €/MWh per p.p. increase in RES penetration & 2012-2019 &             Germany &    RPR \\ \hline
        Halttunen et al. (2021) &   0.482 €/MWh per p.p. increase in RES penetration & 2010-2019 &             Germany &    RPR \\ \hline
\end{tabular}
\end{table}




<br>

### Figures

##### Time Dimension Hyper-Parameters

We'll create a plot showing an example of how regression dates are converted into weightings for the time-series

```python
x = np.linspace(0, 1, 150)
centers = [0.3, 0.5, 0.7]

# Plotting
fig, ax = plt.subplots(dpi=250, figsize=(8, 4))

for center in centers:
    dist = lowess.get_dist(x, center)
    dist_threshold = lowess.get_dist_threshold(dist, frac=0.3)
    weights = lowess.dist_to_weights(dist, dist_threshold)

    ax.plot(x, weights, color='k')
    
x_pos = 0.4
ax.annotate('Interval', xy=(x_pos, 0.95), xytext=(x_pos, 1.00), xycoords='axes fraction', 
            fontsize=6.5, ha='center', va='bottom',
            bbox=dict(boxstyle='square', fc='white'),
            arrowprops=dict(arrowstyle='-[, widthB=7.0, lengthB=1.5', lw=1.0))
    
x_pos = 0.5
ax.annotate('Bandwidth', xy=(x_pos, 0.06), xytext=(x_pos, 0.11), xycoords='axes fraction', 
            fontsize=9.5, ha='center', va='bottom',
            bbox=dict(boxstyle='square', fc='white'),
            arrowprops=dict(arrowstyle='-[, widthB=7.0, lengthB=1.5', lw=1.0))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.1)
eda.hide_spines(ax)
ax.set_xlabel('Data Fraction')
ax.set_ylabel('Relative Weighting')
```




    Text(0, 0.5, 'Relative Weighting')




![png](./img/nbs/output_38_1.png)

