# Outline

> In the meantime I wonder if among the actions / comments below you can focus on preparing a bullet point structure for the contents of the literature review section which you can send out to both Aidan and myself so we can agree on it before you put further work. Do you think it would help?

<br>

### Current paragraph breakdown

* Outlines what the perfect MOE calculation would look like, then explains why we can't calculate it
* Explains the general framework used by MOE researchers, i.e. backcasting electricity prices with a RES generation sensitivity analysis
* Gives an idea of the breadth of markets MOE has been researched in and why Germany will be used to evaluate the tool
* Talks about why regressing against just RES generation won't work, doesn't talk about why RES % penetration isn't a good regressor 
* First half talks about calculation of the marginal MOE between SPs. The second half talks about the need to adapt the price curve estimate over time and emphasises the benefits
* Discusses the benefits of having different models for different batches of the data. I should try and make this paragraph a primer for the more general idea of splitting up the state-space and fitting local models.
* Further discussion around model coefs evolving over time in the context of a Kalman filter approach which looked at diurnal patterns in MOE.
* Talks about an alternative approach to MOE through simulation rather than back-casting on historical data. Could expand more on this topic, there's some good discussion points in the Dec 20 Imperial report.
* Focuses on how the literature mainly uses linear models, goes into detail on a scenario where this leads to poor estimation of the MOE
* Introduces LOWESS, explains the benefits and provides examples from around the MOE context
* Bullet point list of limitations to existing approaches
* Clarifies the model that will be used, begins to introduce the next section

<br>

### General Comments

* Currently I don't provide any £/MWh or EUR/MWh values from the literature
* Currently I don't talk about how using RES % penetration exaggerates the MOE during periods of low demand
* Currently I don't use the word non-parametric anywhere in the text, might be helpful 
* Would be useful to include results on back-casting model accuracy that's reported in the literature
* Should mention somewhere that it's important to calc MOE for the combined RES (then disaggregate after) rather than modelling them separately
* "structural models deliver counterfactual predictions" - UCL Blundell, https://www.ucl.ac.uk/~uctp39a/Manuscript_Blundell_AEA_PP_Jan_13.pdf
* For the LOWESS window I should discuss in terms of the bandwidth
* Should highlight the non-linear nature of the price/load curve as found by Karakstani and Bunn (2008).
* Whilst the MOE manifests itself as a form of price cannibilisation for the RES generators, to the other generators it creates a 'missing-money problem'. In the long-term the saving should be equal to the displaced OPEX (from fuel), in the short-term it should be the displaced OPEX & CAPEX.
* Behind-the-meter generation, which makes up a large share of the solar generation considered in this research, should ideally be considered as a left-ward shift of the demand curve. However, under the standard assumption of demand being perfectly inelastic !!!REF!!! it is equivalent to account for all generation as a right-ward shift of the supply curve.
* In the discussion need to talk about the short term MOE likely being an over-estimate
* Can view the temporal smoothing as a variant of Conditional expectation sampling
* Why doesn't anyone using tne RPR approach use load-RES as a regressor instead of just load

<br>

### Potential Snippets for Key Points

* At its core the MOE is based around the theory that intermittent RES has a marginal cost of zero, meaning that a change in volume of intermittent RES can be modelled as a left/right-ward shift of the supply curve (ceteris parabus). Despite this the majority of the literature does not model this left/right-ward shift, instead focusing on reduced form regression models where RES generation is treated as an exogenous variable.

<br>

### Papers

<b>Literature reviews that have been used to seed this:</b>
* 10.1016/j.eneco.2017.08.003
* !!!Should use!!! 10.1016/j.enpol.2017.04.034
* !!!Should use!!! table 1 - https://www.sciencedirect.com/science/article/pii/S1364032115013866

##### Sensfuss et al. (2008)
- MOE: 7.83 €/MWh
- Year(s): 2006
- Location: Germany
- Method: agent-based model
- DOI: 10.1016/j.enpol.2008.03.035
- Journal: Energy Policy

##### Weigt (2009)
- MOE: 10 €/MWh
- Year(s): 2006-2008
- Location: Germany
- Method: plant-level simulation
- DOI: 10.1016/j.apenergy.2008.11.031
- Journal: Applied Energy

##### Keles et al. (2013)
- MOE: 5.90 €/MWh (peaking to 130 €/MWh)
- Year(s): 2006–2009
- Location: Germany
- Method: econometric regression model including RES
- DOI: 10.1016/j.enpol.2013.03.028
- Journal: Energy Policy

##### Mulder and Scholtens (2013)
- MOE: 0.03% reduction in price for every p.p. increase in wind speeds
- Year(s): 2006–2011, 
- Location: Germany & Netherlands (result is for Germany)
- Method: econometric regression model including RES
- DOI: 10.1016/j.renene.2013.01.025
- Journal: Renewable Energy

##### Tveten et al. (2013)
- MOE: 7.4 €/MWh in 2009 to 3.1 €/MWh in 2011 (solar)
- Year(s): 2006-2011, 
- Location: Germany
- Method: econometric regression model including RES
- DOI: 10.1016/j.enpol.2013.05.060
- Journal: Energy Policy
- Notes: They look at the influence on the clean-spark/dark spreads instead of price directly

##### Wurzburg et al. (2013)
- MOE: 2% reduction in elec price
- Year(s): 2010-2012, 
- Location: Germany & Austria (combined)
- Method: econometric regression model looking at RES
- DOI: 10.1016/j.eneco.2013.09.011
- Journal: Energy Economics

##### Cludius et al. (2014)
- MOE: 6 €/MWh rising to 10 €/MWh
- Year(s): 2010-2012, 
- Location: Germany
- Method: econometric regression model including RES
- DOI: 10.1016/j.eneco.2014.04.020
- Journal: Energy Economics

##### Ketterer (2014)
- MOE: 0.1-1.46% price reduction per p.p. of wind gen
- Year(s): 2006-2012, 
- Location: Germany
- Method: econometric regression model including RES
- DOI: 10.1016/j.eneco.2014.04.003
- Journal: Energy Economics

##### Ederer (2015)
- MOE: 1.3% price reduction for every TWh of wind
- Year(s): 2006-2014, 
- Location: Germany
- Method: structural econometric model (modelling the supply curve shift)
- DOI: 10.1016/j.apenergy.2015.05.033
- Journal: Applied Energy
- Notes: They note that statistical (reduced-form) methods aren't precise enough

##### Kyritsis et al. (2017)
- MOE: ***not quantified directly***
- Year(s): 2010-2015
- Location: Germany
- Method: econometric regression model including RES (GARCH)
- DOI: 10.1016/j.enpol.2016.11.014
- Journal: Energy Policy

##### Bublitz et al. (2017)
- MOE: 1.00 €/MWh in 2014 to 3.30 €/MWh in 2015 (wind ABM), 4.40 €/MWh (wind regression), 2.10 €/MWh (solar ABM), 2.40 €/MWh (solar regression)
- Year(s): 2011-2015
- Location: Germany
- Method: Use both an ABM and reduced-form econometric model
- DOI: 10.1016/j.enpol.2017.04.034
- Journal: Energy Policy
- Notes: Refers to a model in literature as a merit-order model - "Using a fundamental merit-order model of the German electricity system that separates between 34 different power plant types, Weber and Woll (2007)". "The linear regression model has a high explanatory power with R2 ranging from 0.69 to 0.83"

##### de Miera et al. (2008) 
- MOE: 8.6 to 25.1% price reduction
- Year(s): 2005-2007
- Location: Spain
- Method: ***simulation but unclear exactly how***
- DOI: 10.1016/j.enpol.2008.04.022
- Journal: Energy Policy
- Notes: They have a para talking about how there is generally a perception that RES leads to higher electricity bills. "Jensen and Skytte (2003) were the first to point out that a greater RES-E deployment could even reduce final electricity prices". They have a nice plot showing the time-series of the actual price, the forecast price with RES, and the forecast price without RES

##### Gelabert et al. (2011) 
- MOE: 3.7% price reduction
- Year(s): 2005-2012
- Location: Spain
- Method: econometric regression model including RES
- DOI: 10.1016/j.eneco.2011.07.027
- Journal: Energy Economics

##### Ciarreta et al. (2014)
- MOE: between 25 €/MWh and 45 €/MWh for each year
- Year(s): 2008–2012
- Location: Spain
- Method: ***simulation but unclear exactly how***
- DOI: 10.1016/j.enpol.2014.02.025
- Journal: Energy Policy

##### Clo et al. (2015)
- MOE: 2.3 €/MWh for solar and 4.2 €/MWh for wind
- Year(s): 2005–2013
- Location: Italy
- Method: econometric regression model including RES, one version splits up solar and wind
- DOI: 10.1016/j.enpol.2014.11.038
- Journal: Energy Policy

##### Munksgaard and Morthorst (2008)
- MOE: 0.1-0.4 c€/kWh
- Year(s): 2004-2006
- Location: Denmark
- Method: ***seems like reduced-form but unclear***
- DOI: 10.1016/j.enpol.2008.07.024
- Journal: Energy Policy

##### Jonsson et al. (2010)
- MOE: ***Didn't quantify in terms of MOE, instead stated average electricity prices for different wind penetrations***
- Year(s): 2006-2007
- Location: Denmark
- Method: Use a 3D LOWESS fit between price,wind pct share, and the hour of the day. Uses a tri-cube kernel for weighting.
- DOI: 10.1016/j.eneco.2009.10.018
- Journal: Energy Economics
- Notes: Has some interesting discussion around how this technique can also be used for improved modelling of wind-farm strategies through the incorporation of the price-maker effect.

##### Denny et al. (2017) 
- MOE: 3.40 €/MWh per GWh of wind, €44.36 million or 3.76% of total annual dispatch costs in 2009
- Year(s): 2009
- Location: Ireland
- Method: econometric regression model including RES
- DOI: 10.1016/j.renene.2016.11.003
- Journal: Renewable Energy
- Notes: Interesting country due to its high wind penetration

##### Lunackova et al. (2017)
- MOE: 0.7% increase for a 10% increase in solar PV production (!!!Opposite of what we'd expect!!!). -2.5% MOE for wind and hydro.
- Year(s): 2010-2015
- Location: Czech Republic
- Method: econometric regression model including RES
- DOI: 10.1016/j.enpol.2017.02.053
- Journal: Energy Policy

##### Dillig et al. (2016)
- MOE: 5.29 ct/kWh
- Year(s): 2011-2013
- Location: Germany
- Method: They fit a non-linear (dual-exponential) marginal cost curve for dispatchable generation, essentially a parametric version of my approach. They fit a different curve for each year.
- DOI: 10.1016/j.rser.2015.12.003
- Journal: Renewable and Sustainable Energy Reviews
- Notes: States that without RES Germany would have had power shortages. "linear regression models ... do not account for scarcity effects", "prices rise exponentially and linear approximation is responsible for a strong under-estimation of price increases.".

##### McConnell et al. (2013) 
- MOE: 8.6% price reduction, savings of A$1.8 billion (for 5 GW solar)
- Year(s): 2009-2010
- Location: Australia
- Method: Perfect Model! They have access to the underlying bids and offers so they recreate the exact curves for every 5 minutes and simulate the effect of an increase in MC=0 solar
- DOI: 10.1016/j.enpol.2013.01.052
- Journal: Energy Policy

##### Moreno et al. (2012)
- MOE: Found that for every p.p. rise in RES share prices rose by 0.018 p.p (i.e. positive but almost negligible)
- Year(s): 1998–2009
- Location: EU27
- Method: econometric regression model including RES
- DOI: 10.1016/j.energy.2012.06.059
- Journal: Energy
- Notes: They take a panel-data approach but only consider annual data which is likely why they calculate the effect being quite small

##### Woo et al. (2011)
- MOE: $0.32-1.53/MWh per 100 MWh increase in wind generation (per 15 mins)
- Year(s): 2007-2010
- Location: Texas
- Method: econometric regression model including RES
- DOI: 10.1016/j.enpol.2011.03.084
- Journal: Energy Policy
- Notes: They're looking at the BM not DAM

##### Kaufmann and Vaid (2016)
- MOE: $0.26-1.86/MWh for solar, 0.3% emissions reduction
- Year(s): 2010-2012
- Location: Massachusetts 
- Method: econometric regression model including RES
- DOI: 10.1016/j.enpol.2016.03.006
- Journal: Energy Policy
- Notes: They also apply their model to calculate the MOE of carbon as well. When looking at behind-the-meter generation it is called Demand Reduction-Induced Price Effect (DRIPE).

##### Woo et al. (2016)
- MOE: $5.3/MWh per 1GWh of solar, $3.3/MWh for 1GWh of wind
- Year(s): 2012-2015
- Location: California
- Method: econometric regression model including RES
- DOI: 10.1016/j.enpol.2016.02.023
- Journal: Energy Policy

##### Paraschiv et al. (2014)
- MOE: -0.15% per MWh of RES
- Year(s): 2010-2013
- Location: Germany
- Method: econometric regression model including RES
- DOI: 10.1016/j.enpol.2014.05.004
- Journal: Energy Policy
- Notes: Looked at the MOE at different times of day

##### O'Mahoney and Denny (2011)
- MOE: €141M saving, ca.12% reduction
- Year(s): 2009
- Location: Ireland
- Method: econometric regression model including RES
- URL: https://ideas.repec.org/p/pra/mprapa/56043.html

##### Hildmann et al. (2015)
- MOE: 13.4-18.6 (EUR/MWh)
- Year(s): 2011-2013
- Location: Germany and Austria
- Method: Perfect Model! They have access to the underlying bids and offers so they recreate the exact curves for EPEX and simulate wind and solar with MC=0
- DOI: 10.1109/TPWRS.2015.2412376
- Journal: IEEE Transactions on Power Systems
- Notes: They do a sensitivity analysis for different MC costs. "15-min resolution data is averaged to hourly granularity. The loss of the additional information of 15-min data does not affect the quality of the analysis. "

##### Gil et al. (2012)
- MOE: 9.72 EUR/MWh
- Year(s): 2007-2010
- Location: Spain
- Method: LOWESS fit between wind output and price
- DOI: 10.1016/j.enpol.2011.11.067
- Journal: Energy Policy

##### Weber and Woll (2007)
- MOE: 4 EUR/MWh
- Year(s): 2006
- Location: Germany
- Method: ***Need to translate***
- DOI: NA
- URL: https://econpapers.repec.org/paper/duiwpaper/0701.htm
- Journal: NA

##### 
- MOE: 
- Year(s): 
- Location: 
- Method: 
- DOI: 
- Journal: 

<br>

##### Quotes

> In general, detailed assessment of the literature identifies three broad methods that have been used to quantify the merit-order effect. These methods are econometric techniques, power flow/unit commitment techniques and agent-based modelling techniques. Of the three methods, econometric techniques dominate, followed by power flow/unit commitment and then agent-based modelling techniques. - 10.1016/j.eneco.2017.08.003


<br>

### MOE Variants

**Need to go through as many papers as possible fitting them into these categories or making new categories**

Papers that use the same Simulation and Econometric model groupings:
* 10.1016/j.eneco.2013.09.011

Two main families of model, simulation and econometric, are used in the quantification of MOE. Of these the econometric models are used most frequently and can be broken down further into two main types: reduced form regression with RES as an exogenous variable, reduced form regression of dispatchable load with a structural model to incorporate the influence of RES 

PERFECT
* McConnell et al. (2013) 
* Hildmann et al. (2015)
* Ederer (2015)

ESS
* Sensfuss et al. (2008)
* Weigt (2009)
* Bublitz et al. (2017)
* de Miera et al. (2008) 
* Ciarreta et al. (2014)
* Bode and Groscurth (2006) !!! NEED TO ADD IN !!!

RPR
* Keles et al. (2013)
* Mulder and Scholtens (2013)
* Tveten et al. (2013)
* Wurzburg et al. (2013)
* Cludius et al. (2014)
* Ketterer (2014)
* Kyritsis et al. (2017)
* Bublitz et al. (2017)
* Gelabert et al. (2011) 
* Clo et al. (2015)
* Munksgaard and Morthorst (2008)
* Denny et al. (2017) 
* Lunackova et al. (2017)
* Moreno et al. (2012)
* Woo et al. (2011)
* Kaufmann and Vaid (2016)
* Woo et al. (2016)
* Paraschiv et al. (2014)
* O'Mahoney and Denny (2011)
* Jonsson et al. (2010) - 3D LOWESS fit including price and RES %
* Halttunen et al. (2021) !!! NEED TO ADD IN !!!

MSS
* Dillig et al. (2016) - estimates the marginal supply cost curve (same curve as me under our assumptions)
* Weber and Woll (2007) !!! NEED TO ADD IN !!!

!!! Need to specifically address the Staffell paper !!!

<br>

### Proposed paragraph breakdown

* Outline how the idealised way to calculate MOE precisely, then explain why this can't be done. (could put this at the end of the intro to introduce the literature review?)
* Introduce the three main modelling approaches: simulation, reduced-form, structural
* Table showing collated results with information on the modelling approach and market
* Discussion around what simulation models have been used and their advantages/drawbacks. Emphasise that these are mainly used for more qualitative studies.
* Discussion around what reduced-form models have been used and their advantages/drawbacks. Emphasise that this is the most popular approach, in part due to its simplicity.
* Discuss variants to the standard reduced-form appproach: marginal MOE calculation, regressing against % RES penetration instead of power output, etc.
* Discussion around what structural models have been used and their advantages/drawbacks. Emphasise that this forms a sort of best-of-both worlds between the simulation and reduced-form models. Should include mention to the Blundell paper here around how structural models enable counterfactuals to be quantified.
* Focus in on the issues around the constantly evolving market and the need to retrain models/update their coefficients over time. This paragraph should be a primer for the more general idea of splitting up the state-space and fitting local models.
* Highlight that there's a need to train more localised models not just for different time-ranges but also different load-ranges. Discuss the regime-switching models that have been used in this area, should also emphasise the importance of calculating the MOE at the top and bottom of the supply curve.
* Introduce LOWESS, explain the benefits and provides specific examples of scenarios where it would lead to improved modelling of the MOE.
* Summarise the limitations to existing approaches, could use a bullet point list again.
* Clarify the high-level modelling approach that has been identified as optimal based on the literature, then introduce the methodology section.


> Furthermore, as Würzburg et al. (2013) point out, it must be kept in mind that the comparability of studies regarding the merit-order effect is limited due to the heterogeneous approaches, e.g. different sets of included fundamental variables (e.g. fuel prices, market scarcity), alternate scope (inclusion of neighboring countries or emission trading systems) and varying scenarios (no changes or alternative capacity expansion paths).

* ^ Might be worth trying to get some of this in, perhaps before the table of MOE results

Traber et al 2011 -> "In the absence of expanded deployment of renewable energy, a higher price increase of 20% can be expected" this is for 2020 Germany

<br>

### Methodology Outline

##### Current Para Structure

4.1. Data Selection & Processing (Should be quick things to change)
* Where each time-series is sourced from
* Where carbon intensity is sourced from

4.2. LOWESS Model Formulation, Optimisation & Evaluation
* 
* 

4.3. Assumptions & Limitations
* 
* 

##### Bits I want to Change

* Should specifically address the time-of-day aspect
* `Clearly` the next step is to create a multi-variate regression model that includes the traditional variables used in RPR models


* Need to calc the 95% conf and 68% pred intvls

To Do 
- [ ] Add the big literature review table (at the same time check each one has been downloaded with the DOI as the filename)
- [ ] Go over the intro and abstract comments from Paolo (monday morning)
- [ ] Final run over Aidans comments (sunday night)
- [ ] Finish discussion around the MOE estimate (do the lit rev table first)
- [ ] Run the skopt model using the 2 hyper-params (saturday night/sunday morning)
- [ ] Re-run the pred and conf intvl models (!!!not a priority - though pred is higher, could show one pred intvl as heatmap too!!!)
- [ ] Talk about how using RES % penetration exaggerates the MOE during periods of low demand (sunday/monday)
- [x] Add the simple results tables
	- [x] System overview
	- [x] Carbon intensity estimates
	- [x] EPF accuracy metrics
	- [x] MOE and CO2 results
- [x] Add the graphs
	- [x] heatmaps for price and carbon
	- [x] MOE time-series for price and carbon
	- [x] % MOE reduction v % RES
	- [x] Example day with counter-factual price (no longer going to add - could have in the discussion)
- [ ] Add all of the citations into the bib (lit rev tables one then, other ones after monday call)
- [ ] Check for all XXX, REF, \*\*\*, and !!! (before monday call)
- [ ] Check numbering and capitalise tables and figures (before monday call)

* Need to work out how to add in the time complexity element best

Tonight: do the tables and add the graphs, then get skopt running