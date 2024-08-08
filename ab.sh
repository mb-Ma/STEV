# oversampling 
# python main.py common=pyg_pems
# python main.py common=pyg_agcrn_pems

# augmentation
python main.py common=pyg_pems common.augment_data=True
python main.py common=pyg_agcrn_pems common.augment_data=True