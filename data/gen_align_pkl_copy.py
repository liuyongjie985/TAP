#!/usr/bin/env python

import os
import sys
import cPickle as pkl
import numpy

align_path = 'on-align-train/'  # for test.pkl, change 'train' to 'test'
feature_path = 'on-ascii-train/'
outFile = 'align-online-train.pkl'
oupFp_feature = open(outFile, 'wr')

alignment = {}
sentNum = 0
needCount = 19999999999999

scpFile = open('caption/train_caption.txt')
while 1:
    line = scpFile.readline().strip()  # remove the '\n'
    if not line or sentNum >= needCount:
        break
    else:
        key = line.split('\t')[0]
        align_file = align_path + key + '.align'
        print align_file
        with open(align_file) as f_align:
            wordNum = 0
            for align_line in f_align:
                wordNum += 1
        feature_file = feature_path + key + '.ascii'
        fea = numpy.loadtxt(feature_file)
        align = numpy.zeros([fea.shape[0], wordNum], dtype='int8')
        sentNum = sentNum + 1
        penup_index = numpy.where(fea[:, -1] == 1)[0]  # 0 denote pen down, 1 denote pen up
        print penup_index
        with open(align_file) as f_align:
            wordNum = -1
            for align_line in f_align:
                wordNum += 1
                align_tmp = align_line.split()
                print align_tmp
                for i in range(1, len(align_tmp)):
                    print align_tmp[i]
                    pos = int(align_tmp[i])
                    if pos == -1:
                        continue
                    elif pos == 0:
                        align[0:(penup_index[pos] + 1), wordNum] = 1
                    else:
                        align[(penup_index[pos - 1] + 1):(penup_index[pos] + 1), wordNum] = 1
        # numpy.savetxt(check_file, align, fmt='%d')
        for x in range(10):        
            alignment[key+str(x)] = align

        if sentNum / 500 == sentNum * 1.0 / 500:
            print 'process sentences ', sentNum

print 'load align file done. sentence number ', sentNum

pkl.dump(alignment, oupFp_feature)
# pkl.dump(stroke_masks,oupFp_strokeMask)
print 'save file done'
oupFp_feature.close()
# oupFp_strokeMask.close()
