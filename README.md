<p align="center">
 <img width="100px" src="https://github.githubassets.com/images/mona-loading-default.gif" align="center" alt="Logo" />
 <h2 align="center">Beyond Fixed Variables: Expanding-variate Time Series
Forecasting via Flat Scheme and Spatial-temporal Focal Learning</h2>
</p>

# :house: STEV

## TL; DR
We introduce EVTSF, an emerging task that targets a frequently overlooked aspect of CPSs — their evolution through sensing expansion. To address this challenge, we propose STEV, a pioneering framework tailored to the unique demands of EVTSF.

## Data Preparation
We sourced from three public multivariate time series datasets: [Electricity](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014),[PeMS](https://www.kaggle.com/datasets/liuxu77/largest), and [Weather](https://drive.google.com/drive/folders/1sPCg8nMuDa0bAWsHPwskKkPOzaVcBneD). You need to download these datasets first, then run the data_process scripts to obtain the **expanding-variate time series datasets**.

```
# Generating EElectricity

# Generating EPeMS

# Generating EWeather

```

## HOW TO RUN STEV

### Prerequisites
```
pip install -r requirements.txt
```
### Prepare datasets
```
cd ./data/process

# EElectricity
python ./process_elc.py

# EPeMS
run ./process_largest_data.ipynb
python ./process_pems.py

# EWeather
python ./raw_weather.py
python process_weather.py
```

### Run STEV
### Training & Validating & Infering
```
# EElectricity
python main.py common=pyg_cons_gwnet_electricity

# EPeMS
python main.py common=pyg_cons_gwnet_pems

# EWeather
python main.py common=pyg_cons_gwnet_weather

```

### For all experiments
We also provide a script for running all EVTSF models including STEV, Uni methods, FPTM methods. Moreover, the Oracle experiment is included.
```
bash run.sh
```

### Only for Infering with trained weights
```
python infer.py # modify the weight folder path in infer.py
```
