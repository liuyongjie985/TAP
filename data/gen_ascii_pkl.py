<<<<<<< HEAD
#!/usr/bin/env python

import os
import sys
import cPickle as pkl
import numpy as np

FLAG = "test"

feature_path = 'on-ascii-' + FLAG + '/'  # for test.pkl, change 'train' to 'test'
outFile = 'online-' + FLAG + '.pkl'
oupFp_feature = open(outFile, 'wr')

features = {}

sentNum = 0
needCount = 100
scpFile = open(FLAG + '_caption.txt')

while 1:
    line = scpFile.readline().strip()  # remove the '\r\n'
    if not line or sentNum >= needCount:
        break
    else:
        key = line.split('\t')[0]
        feature_file = feature_path + key + '.ascii'
        mat = np.loadtxt(feature_file)
        sentNum = sentNum + 1
        features[key] = mat
        # generate stroke_mask

        # penup_index = np.where(mat[:,-1] == 1)[0] # 0 denote pen down, 1 denote pen up
        # strokeNum = len(penup_index)
        # stroke_mat = np.zeros([strokeNum, mat.shape[0]], dtype='float32')
        # for i in range(0,strokeNum):
        # Mask
        # if i == 0:
        #    stroke_mat[i,0:(penup_index[i]+1)] = 1
        # else:
        # stroke_mat[i,(penup_index[i-1]+1):(penup_index[i]+1)] = 1
        # Index
        # stroke_mat[i,penup_index[i]] = 1
        # normMask
        # if i == 0:
        #    stroke_mat[i,0:(penup_index[i]+1)] = 1 / (penup_index[i]+1)
        # else:
        #    stroke_mat[i,(penup_index[i-1]+1):(penup_index[i]+1)] = 1 / ((penup_index[i]+1) - (penup_index[i-1]+1))
        if sentNum / 500 == sentNum * 1.0 / 500:
            print 'process sentences ', sentNum

print 'load ascii file done. sentence number ', sentNum

pkl.dump(features, oupFp_feature)
print 'save file done'
oupFp_feature.close()
=======
#!/usr/bin/env python

import os
import sys
import cPickle as pkl
import numpy as np
FLAG = "test"
feature_path='on-ascii-'+FLAG+'/' # for test.pkl, change 'train' to 'test'
outFile='online-'+FLAG+'.pkl'
oupFp_feature=open(outFile,'wr')

features={}

sentNum=0

scpFile=open('caption/'+FLAG+'_caption.txt')
while 1:
    line=scpFile.readline().strip() # remove the '\r\n'
    if not line:
        break
    else:
        key = line.split('\t')[0]
        feature_file = feature_path + key + '.ascii'
        mat = np.loadtxt(feature_file)
        sentNum = sentNum + 1
        features[key] = mat
        # generate stroke_mask
        
        #penup_index = np.where(mat[:,-1] == 1)[0] # 0 denote pen down, 1 denote pen up
        #strokeNum = len(penup_index)
        #stroke_mat = np.zeros([strokeNum, mat.shape[0]], dtype='float32')
        #for i in range(0,strokeNum):
            # Mask
            #if i == 0:
            #    stroke_mat[i,0:(penup_index[i]+1)] = 1
            #else:
               # stroke_mat[i,(penup_index[i-1]+1):(penup_index[i]+1)] = 1
            # Index
            #stroke_mat[i,penup_index[i]] = 1
            # normMask
            #if i == 0:
            #    stroke_mat[i,0:(penup_index[i]+1)] = 1 / (penup_index[i]+1)
            #else:
            #    stroke_mat[i,(penup_index[i-1]+1):(penup_index[i]+1)] = 1 / ((penup_index[i]+1) - (penup_index[i-1]+1))
        if sentNum / 500 == sentNum * 1.0 / 500:
            print 'process sentences ', sentNum

print 'load ascii file done. sentence number ',sentNum

pkl.dump(features,oupFp_feature)
print 'save file done'
oupFp_feature.close()
>>>>>>> 5df2f4fb61733b60455b5f86d6869a4bceded076
