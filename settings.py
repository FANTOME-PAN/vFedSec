"""Debug mode"""
DEBUG = False


"""Vertical FL settings"""
# Simulated clients will be instantiated sequentially, with Client ID (CID) from '0' to 'N'
# In this example, CID of the active party is '0'; passive parties are divided into 2 types;
# CIDs of type A passive parties are '1', '2';
# CIDs of type B passive parties are '3', '4'.
ACTIVE_PARTY_CID = '0'
PASSIVE_PARTY_CIDs = {
    'A': ['1', '2'],
    'B': ['3', '4'],
}


"""Training hyper-parameters"""
NUM_ITERATION = 5  # odd
LEARNING_RATE = 0.01
BATCH_SIZE = 256


"""Paths"""
# Please specify the path to the data of each party.
CID_TO_DATA_PATH = {
    str(cid): f'data/mimic3/p{cid}_data.csv' for cid in range(5)
}
ACTIVE_PARTY_LOCAL_MODULE_SAVE_PATH = 'models/active_party_local_module.pth'
PASSIVE_PARTY_LOCAL_MODULE_SAVE_PATH_FORMAT = 'models/passive_party_local_module_%s.pth'
GLOBAL_MODULE_SAVE_PATH = 'models/global_module.pth'
# The path of the pretrained model for the active party.
# Keep this field empty if no pretraining.
ACTIVE_PARTY_PRETRAINED_MODEL_PATH = None


'''Please do not change the following fields if not necessary'''
# num_rounds = 1 + 1.5N + 0.5 = 1.5(N+1)
TRAINING_ROUNDS = int((NUM_ITERATION + 1) * 3 // 2)
CID_TO_TYPE = {cid: t for t, cids in PASSIVE_PARTY_CIDs.items() for cid in cids}
INDEX_TO_TYPE = tuple(PASSIVE_PARTY_CIDs.keys())
# TYPE_TO_INDEX = {t: idx for t, idx in zip(PASSIVE_PARTY_CIDs.keys(), range(len(PASSIVE_PARTY_CIDs)))}
TYPE_TO_INDEX = {t: idx for idx, t in enumerate(INDEX_TO_TYPE)}
CLIP_RANGE = 64
TARGET_RANGE = 1 << 27
ENABLE_PROFILER = True
