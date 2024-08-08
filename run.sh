##### Oracle experiment
# python main.py common=gwnet_oracle_electricity
# python main.py common=gwnet_oracle_pems
# python main.py common=gwnet_oracle_weather
# python main.py common=agcrn_oracle_electricity
# python main.py common=agcrn_oracle_pems
# python main.py common=agcrn_oracle_weather
# python main.py common=iTransformer_oracle_weather
# python main.py common=iTransformer_oracle_electricity
# python main.py common=mega_oracle_weather
# python main.py common=mega_oracle_electricity
# python main.py common=sgp_oracle_pems
# python main.py common=sgp_oracle_weather
# python main.py common=msgnet_oracle_electricity


### masking experiment
# python main.py common=gwnet_mask_weather
# python main.py common=gwnet_mask_pems
# python main.py common=gwnet_mask_electricity
# python main.py common=agcrn_mask_weather
python main.py common=agcrn_mask_pems
# python main.py common=agcrn_mask_electricity
# python main.py common=iTransformer_mask_pems
# python main.py common=iTransformer_mask_weather
# python main.py common=iTransformer_mask_electricity
# python main.py common=mega_mask_electricity
# python main.py common=mega_mask_weather
# python main.py common=msgnet_mask_electricity
# python main.py common=msgnet_mask_weather

### pyg experiment
# python main.py common=pyg_pems
# python main.py common=pyg_electricity
# python main.py common=pyg_weather

# pyg agcrn experiment
# python main.py common=pyg_agcrn_electricity
# python main.py common=pyg_agcrn_weather

# cons experiments
# python main.py common=pyg_cons_agcrn_pems
# python main.py common=pyg_cons_agcrn_electricity
# python main.py common=pyg_cons_agcrn_weather
# python main.py common=pyg_cons_gwnet_pems
# python main.py common=pyg_cons_gwnet_electricity
# python main.py common=pyg_cons_gwnet_electricity common.weight=0.1
# python main.py common=pyg_cons_gwnet_weather common.weight=0.01
# python main.py common=pyg_cons_gwnet_weather common.weight=0.05



# TESTING
# python main.py common=uni_electricity common.augementation_feat=0 common.augementation_rate=1 common.epochs=1
# python main.py common=uni_pems common.augementation_feat=0 common.augementation_rate=1 common.epochs=1
