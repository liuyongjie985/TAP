import os
import json
import sys
import numpy as np

DATAPATH = "./book66"

files = os.listdir(DATAPATH)

for file in files:
    f = DATAPATH + "/" + file
    with open(f, 'r') as load_f:
        load_dict = json.load(load_f)

        for x, y in load_dict.items():
            print(y)

        with open("./output.json", 'w') as  f:
            json.dump(load_dict, f)
            break
