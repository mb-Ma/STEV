### FLAT, AUGMENTATION
# python main.py common=pyg_pems common.augment_data=True
# python main.py common=pyg_agcrn_pems common.augment_data=True

### FLAT, oversampling
# python main.py common=pyg_pems common.is_over_sampling=True
# python main.py common=pyg_agcrn_pems common.is_over_sampling=True common

# python main.py common=pyg_cons_gwnet_pems common.weight=1.0 # no focal weights
# python main.py common=pyg_cons_gwnet_pems common.data_path=data/processed/spatial_pems-63_296-3_447-2-12_12_C.npz
# python main.py common=pyg_cons_gwnet_pems common.data_path=data/processed/random_pems-63_296-3_447-2-12_12_C.npz
# python main.py common=gwnet_oracle_pems common.data_path=data/processed/spatial_pems-63_447-3_447-2-12_12_C_Oracle.npz
# python main.py common=gwnet_oracle_pems common.data_path=data/processed/random_pems-63_447-3_447-2-12_12_C_Oracle.npz
# python main.py common=gwnet_mask_pems common.data_path=data/processed/spatial_pems-63_296-3_447-2-12_12_C.npz
# python main.py common=gwnet_mask_pems common.data_path=data/processed/random_pems-63_296-3_447-2-12_12_C.npz


# python main.py common=pyg_cons_gwnet_pems common.weight=0.1
python main.py common=pyg_cons_gwnet_pems common.weight=0.05
# python main.py common=pyg_cons_gwnet_pems common.weight=0.7


