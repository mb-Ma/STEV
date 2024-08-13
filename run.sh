### STEV experiment
python main.py common=pyg_cons_gwnet_pems
python main.py common=pyg_cons_gwnet_electricity
python main.py common=pyg_cons_gwnet_weather


### Oracle experiment
python main.py common=gwnet_oracle_electricity
python main.py common=gwnet_oracle_pems
python main.py common=gwnet_oracle_weather
python main.py common=agcrn_oracle_electricity
python main.py common=agcrn_oracle_pems
python main.py common=agcrn_oracle_weather
python main.py common=iTransformer_oracle_weather
python main.py common=iTransformer_oracle_electricity
python main.py common=msgnet_oracle_weather
python main.py common=msgnet_oracle_electricity
python main.py common=msgnet_oracle_pems

### masking experiment
python main.py common=gwnet_mask_weather
python main.py common=gwnet_mask_pems
python main.py common=gwnet_mask_electricity
python main.py common=agcrn_mask_weather
python main.py common=agcrn_mask_pems
python main.py common=agcrn_mask_electricity
python main.py common=iTransformer_mask_pems
python main.py common=iTransformer_mask_weather
python main.py common=iTransformer_mask_electricity
python main.py common=msgnet_mask_electricity
python main.py common=msgnet_mask_weather
python main.py common=msgnet_mask_pems

### UNI experiment
python main.py common=uni_electricity common.augementation_feat=0 common.augementation_rate=1 common.epochs=1
python main.py common=uni_pems common.augementation_feat=0 common.augementation_rate=1 common.epochs=1
