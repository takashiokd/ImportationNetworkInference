# Inference of Importation Network
Code for reproducing analyses in the manuscript:

Takashi Okada, Giulio Isacchini, QinQin Yu, and Oskar Hallatschek

# Description of Repository

In this study, we developed a method for inferring the importation rates, i.e., the proportion of infections that a community imports from other communities, based on how rapidly the allele frequencies in the focal community converge to those in the donor communities.

### usage_HMM_WF.ipynb
The HMM-EM method is demonstrated using simulated data.

* INPUT
    * Spatio-temporal data of allele (or lineage) counts.
    * Spatio-temporal data of the total number of sampled sequences.

* OUTPUT
    * Record of log likelihood across EM cycles.
    * Inferred importation-rate matrix ${\mathbf A}_{ij}$.
    * Inferred effective population size.
    * Least squares estimation of ${\mathbf A}_{ij}$.(noise ignored).
    * Inferred measurement noise overdispersion.

### modules/
Miscellaneous tools used for analysis.

### data/
Data from England and the USA used in the analysis.

### data/
Data from England and USA used in analysis. 


 
