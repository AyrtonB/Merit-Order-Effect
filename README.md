[![DOI](https://zenodo.org/badge/326810654.svg)](https://zenodo.org/badge/latestdoi/326810654) [![PyPI version](https://badge.fury.io/py/moepy.svg)](https://badge.fury.io/py/moepy) [![Binder](https://notebooks.gesis.org/binder/badge_logo.svg)](https://notebooks.gesis.org/binder/v2/gh/AyrtonB/Merit-Order-Effect/main?urlpath=lab)

# MOE-Py

This repository outlines the development and usage of code and analysis used in calculating the Merit-Order-Effect (MOE) of renewables on price and carbon intensity of electricity markets. Beyond MOE analysis the `moepy` library can be used more generally for standard, quantile, and bootstrapped LOWESS estimation. The particular implementation of LOWESS in this software has been extended to significantly reduce the computational resource required. 

To reduce dependency bloat the `moepy` library can be installed in two ways. If you just wish to use the LOWESS curve fitting aspects of the library then you can install it using:

```bash
pip install moepy
```

If you wish to reproduce the analysis used for estimation of the Merit Order Effect please run :

```bash
pip install moepy[research]
```

`moepy` makes it simple to fit a LOWESS curve, a quick start example to generate the plot below can be found [here](https://ayrtonb.github.io/Merit-Order-Effect/ug-04-gb-mcc/).

![](img/latest_gb_mcc.png)

<br>

The library also includes the option to ensemble LOWESS models together and smooth them over time, an example is shown below for the marginal cost curve of dispatchable generation in Great Britain. For ease of use this has been made to work directly with Pandas datetime indexes and is exposed through a standard sklearn model API.

![](img/UK_price_MOE_heatmap.png)

<br>

### Paper

The `moepy` library was developed to enable new research into the Merit-Order-Effect of renewables in the British and German power systems. The full paper can be found [here](https://ayrtonb.github.io/Merit-Order-Effect/assets/Quantifying%20the%20MOE%20in%20Britain%20&%20Germany.pdf), the abstract is shown below:

> This paper presents an empirical analysis of the reduction in day-ahead market prices and CO<sub>2</sub> emissions due to increased renewable generation on both the British and German electricity markets. This Merit Order Effect is becoming more important as markets evolve to incorporate greater shares of renewable energy sources, driving renewable capture price cannibilisation and market volatility. However, explicitly determining its magnitude can be challenging due to the confidential nature of the data required. Existing statistical methods for inferring this effect have focused on linear parametric approaches. However, these have a number of disadvantages. In this work we propose a flexible non-parametric blended Locally Weighted Scatterplot Smoothing approach  that captures the non-linear relationship between electricity price and dispatchable generation. This is the first application of this method in this context. We found the accuracy of this approach comparable to methods used in modern price back-casting literature. Our results indicate that the Merit Order Effect has increased dramatically over the time period analysed, with a sharp and continuing increase from 2016 in Britain. We found that renewables delivered total reductions equal to 318M and 442M tonnes of CO<sub>2</sub>  and savings of €56B and £17B in Germany and Britain respectively.

The key premise behind the analysis is that intermittent renewables with no fuel costs displace high-cost dispatchable generation - this is called the Merit Order Effect (MOE). The effect can be visualised as a rightward shift in the marginal price curve of electricity, which combined with the inelasticity of demand results in a lower market clearing price (shown below).

![](img/MOE_diagram_supply_shift.png)

In this work a time-adaptive LOWESS was used to estimate the marginal price curve, then simulate the MOE. We calculated significant CO2 emission and electricity price savings for Britain and Germany, results for 2019 are shown in the table below.

|                       |   Germany |   Great Britain |
|:----------------------|----------:|----------------:|
| Price ([EUR,GBP]/MWh) |     20.53 |            9.8  |
| Price Reduction (%)   |     36.7  |           19.3  |
| Carbon (Tonnes/h)     |      5085 |           1637  |
| Carbon Reduction (%)  |     34.88 |           33.53 |

We identified a strong relationship between increasing renewable penetration and the Merit-Order-Effect. In Britain the MOE has seen a sharp increase since 2016, with an average 0.67% price reduction per percentage point increase in renewable penetration.

![](img/GB_MOE_RES_relationship_95_CI.png)

<br>

### Examples

Several notebooks have been created to show examples of how LOWESS estimations can be made using various data sources and for different analysis purposes, these include:
* Quantile estimation of hydro-power production in Portgual
* Confidence interval estimation of gravitational wave observations
* Cleaning of wind power curves
* Estimation of electricity price curves

Key plots from each of these can be seen below.

![](img/lowess_fit_examples.png)

If you have used `moepy` for something cool and want to share it with others please create a pull request containing a notebook with your self-contained example.

<br>

### Referencing

If you use this software please cite it using the following:

```
@software{bourn_moepy_2021,
    title = {moepy},
    url = {https://ayrtonb.github.io/Merit-Order-Effect/},
    abstract = {This repository outlines the development and usage of code and analysis used in calculating the Merit-Order-Effect (MOE) of renewables on price and carbon intensity of electricity markets. Beyond MOE analysis the `moepy` library can be used more generally for standard, quantile, and bootstrapped LOWESS estimation. The particular implementation of LOWESS in this software has been extended to significantly reduce the computational resource required.},
    author = {Bourn, Ayrton},
    month = mar,
    year = {2021},
    doi = {10.5281/zenodo.4642896},
}
```