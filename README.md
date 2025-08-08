<p align="center">
 <img width="100px" src="https://github.githubassets.com/images/mona-loading-default.gif" align="center" alt="Logo" />
 <h2 align="center">Beyond Fixed Variables: Expanding-variate Time Series
Forecasting via Flat Scheme and Spatial-temporal Focal Learning</h2>
</p>

# :house: STEV

## TL; DR
We introduce an emerging task, **EVTSF**, addressing a frequently overlooked aspect of CPSs, that is evolving with sensing expansion. In response, we present **STEV**, a pioneering framework designed to tackle the unique challenges of EVTSF.

## Data
This work sources from three public datasets: [EElectricity](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014),[EPeMS](https://www.kaggle.com/datasets/liuxu77/largest), and [EWeather](https://drive.google.com/drive/folders/1sPCg8nMuDa0bAWsHPwskKkPOzaVcBneD). 

you can find these datasets from the published work through the provided URL.
Notably, for the EPeMS dataset, you have to run two processing scripts. The first is to get the filter sensors and time ranges, 
then you can obtain the data for training, validating, and testing.

The Expanding-variant Time Series datasets are processed by the scripts (you can find them in the folder "./data/scripts/").

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
