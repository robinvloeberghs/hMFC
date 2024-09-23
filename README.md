# hMFC
Repository for the Hierarchical Model for Fluctuations in Criterion (hMFC), a hierarchical Bayesian framework that allows the estimation of trial-by-trial fluctuations in decision criterion.

Paper: A Bayesian Hierarchical Model of Trial-To-Trial Fluctuations in Decision Criterion, bioRxiv https://doi.org/10.1101/2024.07.30.605869.

If you use this code, please cite the paper.

Contact: Robin Vloeberghs (robin.vloeberghs@kuleuven.be) / Scott Linderman (scott.linderman@stanford.edu).



---

#### Installation
```bash
1. Install Anaconda using following link
https://docs.anaconda.com/anaconda/install/

2. Open Anaconda Prompt

3. Create new conda environment and activate it
conda create -n hmfc python=3.10
conda activate hmfc

4. Install packages
conda install pip==24.2
pip install equinox==0.11.7
pip install seaborn==0.13.2
pip install matplotlib==3.9.2

5. Install dynamax
git clone https://github.com/probml/dynamax
cd dynamax
pip install -e.

```

#### Getting started

In your favorite Python IDE (e.g. Spyder):
```python
conda activate hmfc
spyder
```



