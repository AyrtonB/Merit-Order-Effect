[![DOI](https://zenodo.org/badge/326810654.svg)](https://zenodo.org/badge/latestdoi/326810654) [![PyPI version](https://badge.fury.io/py/moepy.svg)](https://badge.fury.io/py/moepy)

# Merit-Order-Effect

This repository/site outlines the development and usage of code and analysis used in calculating the Merit-Order-Effect (MOE) of renewables on price and carbon intensity of electricity markets. Beyond MOE analysis the `moepy` library can be used more generally for standard, quantile, and bootstrapped LOWESS estimation. The particular implementation of LOWESS in this software has been extended to significantly reduce the computational resource required.

<br>

### The Paper

The `moepy` library was developed to enable new research into the Merit-Order-Effect of renewables in the British and German power systems. The full paper can be found [here](#)(<b>this will be made available once the paper has been submitted</b>), the abstract is shown below:

> This paper presents an empirical analysis of the reduction in day-ahead market prices and CO$_{2}$ emissions due to increased renewable generation on both the British and German electricity markets. This research aim is delivered through ex-post analysis of the Merit Order Effect (MOE) using a hybrid statistical/simulation approach.
> 
> Existing research focuses on linear methods for modelling the merit order stack, masking the larger MOE seen in the steeper top/bottom regions. In this work a blended LOWESS model is used to capture the non-linear relationship between electricity price and dispatchable generation, with historical renewable output data then used to simulate the MOE. The stationary nature of many existing methodologies means they struggle to adapt to changes in the system such as the effect of the Covid-19 pandemic, we use a time-adaptive model to effectively address this limitation. Alongside an extension to the standard LOWESS implementation the use of a time-adaptive model significantly reduces the computational resource required. 
> 
> Our results indicate that renewables delivered reductions equal to 318M tonnes of CO$_{2}$ between 2010 and 2020, and 56B EUR between 2015 and 2020 in Germany. In GB the reductions amounted to a 442M tonnes of CO$_{2}$and £17B saving between 2010 and 2020.

<br>

### Repo Publishing - To Do

Notebook Polishing Changes:
- [x] Add docstrings (can be one-liners unless shown in the user-guides or likely to be used often)
- [x] Add a mini sentence or two at the top of each nb explaining what it's about
- [x] Ensure there is a short explanation above each code block
- [x] Move input data to a raw dir
- [ ] Check all module imports are included in settings.ini
- [x] Re-run all of the notebooks at the end to check that everything works sequentially

Additional Code:
- [x] Re-attempt LIGO fitting example as part of a user-guide
- [ ] Add in the prediction and confidence interval plots
- [ ] Add a lot more to the EDA examples
- [ ] Every week re-run a single analysis (could be based on the example in the user-guide) and show the generated fit at the top of the ReadMe
- [ ] Try to speed things up, e.g. with Numba ([one person has already started doing this](https://gist.github.com/agramfort/850437#gistcomment-3437320))
- [ ] Get the models saved on S3 or figshare and pulled into binder via a postBuild script

External/ReadMe
- [ ] Separate the binder and development `environment.yml` files (have the dev one inside the batch scripts folder)
- [x] Add the GH action for version assignment triggering pypi push and zenodo update
- [ ] Just before the paper is published set the version to 1.0.0 and have a specific Binder link that builds from that version as stored in the Zenodo archive ([example guide here](https://blog.jupyter.org/binder-with-zenodo-af68ed6648a6))
- [ ] Link the zotero collection
- [ ] Add citations for both the external data I use and the resulting time-series I generate
- [ ] Add bibtex citation examples for both the paper and the code (could use [this](https://citation-file-format.github.io/cff-initializer-javascript/))
- [ ] Mention the new module in the [gist](https://gist.github.com/agramfort/850437) that some of the basic regression code was inspired by 
