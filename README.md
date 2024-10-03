# hMFC
Repository for the Hierarchical Model for Fluctuations in Criterion (hMFC), a hierarchical Bayesian framework that allows the estimation of trial-by-trial fluctuations in decision criterion.

Paper: A Bayesian Hierarchical Model of Trial-To-Trial Fluctuations in Decision Criterion, bioRxiv https://doi.org/10.1101/2024.07.30.605869.

If you use this code, please cite the paper.

Contact: Robin Vloeberghs (robin.vloeberghs@kuleuven.be)



---

#### Installation

This model is written in Python, so you'll need to have a program to run Python code.

**1. Install Anaconda**
   * Download and install Anaconda by following the instructions provided at this link: [Anaconda Installation Guide](https://docs.anaconda.com/anaconda/install/)

**2. Open Anaconda Prompt**
   * Launch Anaconda Prompt from your Start menu or application launcher.
   
**3. Create and Activate a New Conda Environment** 
   * Create a new conda environment named `hmfc` with Python 3.10:
```bash
conda create -n hmfc python=3.10
```
   * And then activate the environment:
```bash
conda activate hmfc
```
**4. Install Required Packages**
   * Install the required packages using the following commands:
```bash
conda install pip==24.2
```
pip install equinox==0.11.7 seaborn==0.13.2 matplotlib==3.9.2 dill==0.3.8
```
**5. Install Dynamax**
   * Clone the Dynamax repository:
```bash
git clone https://github.com/probml/dynamax
cd dynamax
```
   * And install it:
```bash
pip install -e.
```

#### Getting started
Download the code of the hMFC repo
```bash
cd ..
git clone https://github.com/robinvloeberghs/hMFC
cd hMFC
```

Open your favorite Python IDE (for example Spyder):
```python
conda activate hmfc
spyder
```



