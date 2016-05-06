import os
import numpy as np 

DATA_DIR = '~/dataset/CASIA_maxpy_clean'
LST_DIT = './data'

to_write_train = []
to_write_val = []
n = 0
people = os.listdir(DATA_DIR)
for p_idx in len(people):
    p = people[p_idx]
    pics = os.listdir(os.path.join(DATA_DIR, p))
    rnd_perm = np.random.permutation(len(pics))
    train_idx = rnd_perm[len(pics)/4:]
    val_idx = rnd_perm[:len(pics)/4]
    for i in train_idx:
        n += 1
        to_write_train.append('%d\t%d\t%s\n'%(n, p_idx, os.path.join(p, pics[i])))
    for i in val_idx:
        n += 1
        to_write_val.append('%d\t%d\t%s\n'%(n, p_idx, os.path.join(p, pics[i])))

np.random.shuffle(to_write_train)
np.random.shuffle(to_write_val)

with open(os.path.join(LST_DIR, 'train.lst'), 'w') as f:
    for line in to_write_train:
        f.write(line)
with open(os.path.join(LST_DIR, 'val.lst'), 'w') as f:
    for line in to_write_val:
        f.write(line)
