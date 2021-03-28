[![DOI](https://zenodo.org/badge/326810654.svg)](https://zenodo.org/badge/latestdoi/326810654) [![PyPI version](https://badge.fury.io/py/moepy.svg)](https://badge.fury.io/py/moepy)

# Merit-Order-Effect

This repository/site outlines the development and usage of code and analysis used in calculating the Merit-Order-Effect (MOE) of renewables on price and carbon intensity of electricity markets. Beyond MOE analysis the `moepy` library can be used more generally for standard, quantile, and bootstrapped LOWESS estimation. The particular implementation of LOWESS in this software has been extended to significantly reduce the computational resource required.

You can install the library using:

```bash
pip install moepy
```

<br>

### The Paper

The `moepy` library was developed to enable new research into the Merit-Order-Effect of renewables in the British and German power systems. The full paper can be found [here](#)[^1], the abstract is shown below:

[^1]: This will be made available once the paper has been submitted.

> This paper presents an empirical analysis of the reduction in day-ahead market prices and CO$_{2}$ emissions due to increased renewable generation on both the British and German electricity markets. This research aim is delivered through ex-post analysis of the Merit Order Effect (MOE) using a hybrid statistical/simulation approach.
> 
> Existing research focuses on linear methods for modelling the merit order stack, masking the larger MOE seen in the steeper top/bottom regions. In this work a blended LOWESS model is used to capture the non-linear relationship between electricity price and dispatchable generation, with historical renewable output data then used to simulate the MOE. The stationary nature of many existing methodologies means they struggle to adapt to changes in the system such as the effect of the Covid-19 pandemic, we use a time-adaptive model to effectively address this limitation. Alongside an extension to the standard LOWESS implementation the use of a time-adaptive model significantly reduces the computational resource required. 
> 
> Our results indicate that renewables delivered reductions equal to 318M tonnes of CO$_{2}$ between 2010 and 2020, and 56B EUR between 2015 and 2020 in Germany. In GB the reductions amounted to a 442M tonnes of CO$_{2}$and £17B saving between 2010 and 2020.