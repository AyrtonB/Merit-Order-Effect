[![DOI](https://zenodo.org/badge/326810654.svg)](https://zenodo.org/badge/latestdoi/326810654) [![PyPI version](https://badge.fury.io/py/moepy.svg)](https://badge.fury.io/py/moepy)

# Merit-Order-Effect

Code and analysis used for calculating the merit order effect of renewables on price and carbon intensity of electricity markets

<br>

### Repo Publishing - To Do

Notebook Polishing Changes:
- [x] Add docstrings (can be one-liners unless shown in the user-guides or likely to be used often)
- [x] Add a mini sentence or two at the top of each nb explaining what it's about
- [x] Ensure there is a short explanation above each code block
- [x] Move input data to a raw dir
- [ ] Check all module imports are included in settings.ini
- [x] Re-run all of the notebooks at the end to check that everything works sequentially

Completed Notebooks:
- [x] Retrieval
- [x] EDA
- [x] LOWESS (start with the biggy)
- [x] Price Surface Estimation
- [x] Price MOE
- [x] Carbon Surface Estimation and MOE
- [x] Prediction and Confidence Intervals
- [x] Hyper-Parameter Tuning
- [x] Tables and Figures

New Code:
- [ ] Separate the binder and development `environment.yml` files
- [ ] Re-attempt LIGO fitting example as part of a user-guide
- [ ] Add in the prediction and confidence interval plots
- [ ] Add a lot more to the EDA examples
- [ ] Every week re-run a single analysis (could be in the user-guide) and show the generated fit at the top of the ReadMe
- [ ] Try to speed things up, e.g. with Numba ([one person has already started doing this](https://gist.github.com/agramfort/850437#gistcomment-3437320))
- [ ] Get the models saved on S3 or figshare and pulled into binder via a postBuild script

External/ReadMe
- [x] Add the GH action for version assignment triggering pypi push and zenodo update
- [ ] Just before the paper is published set the version to 1.0.0 and have a specific Binder link that builds from that version as stored in the Zenodo archive
- [ ] Could link the zotero collection
- [ ] Add citations for both the external data I use and the resulting time-series I generate
- [ ] Add bibtex citation examples for both the paper and the code (could use [this](https://citation-file-format.github.io/cff-initializer-javascript/))
- [ ] Publish the latest version to PyPi
- [ ] Mention the new module in the [gist](https://gist.github.com/agramfort/850437) that some of the basic regression code was inspired by 
