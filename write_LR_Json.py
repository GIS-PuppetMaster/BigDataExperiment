import pickle

with open('checkPoint_LR.json', 'wb') as f:
    lr = 0.005
    pickle.dump(lr,f)