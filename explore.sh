# explore random and spatial expansion

# oracle
# python main.py common=gwnet_oracle_pems common.data_path=data/processed/random_pems-63_447-3_447-2-12_12_C_Oracle.npz
# python main.py common=gwnet_oracle_pems common.data_path=data/processed/spatial_pems-63_447-3_447-2-12_12_C_Oracle.npz

# FPTM
python main.py common=gwnet_mask_pems common.data_path=data/processed/random_pems-63_296-3_447-2-12_12_C.npz
python main.py common=gwnet_mask_pems common.data_path=data/processed/spatial_pems-63_296-3_447-2-12_12_C.npz