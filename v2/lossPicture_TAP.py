'''
将CROHME_TAP_%d.log转化为图片

'''

import matplotlib.pyplot as plt  # 约定俗成的写法plt
# 首先定义两个函数（正弦&余弦）
import re

INPUT_FILE = "CROHME_TAP_2.log"
OUTPUT_FILE = "loss_pic_TAP_2.png"
# -π to +π 的256个值

file = open(INPUT_FILE)
X = []
Y = []
wer_Y = []
sacc_Y = []
line_num = 0
now_length = 0

while 1:
    lines = file.readlines(100000)
    if not lines:
        break
    for line in lines:
        matchObj = re.match(r"Epoch  (?P<epoch>\d+) Update  (?P<update>\d+) cost  (?P<cost>\d+\.\d+?)", line,
                            re.M | re.I)

        if matchObj:
            print("matchObj.group() : ", matchObj.group("epoch"))
            print("matchObj.group(1) : ", matchObj.group("update"))
            print("matchObj.group(2) : ", matchObj.group("cost"))
            X.append(int(matchObj.group("update")) / 1100)
            Y.append(float(matchObj.group("cost")))
            now_length += 1
        else:
            pass

        matchObj = re.match(r"Valid WER: (?P<wer>\d+\.\d+)%, SACC: (?P<sacc>\d+\.\d+)%", line,
                            re.M | re.I)
        if matchObj:

            print("matchObj.group() : ", matchObj.group("wer"))
            print("matchObj.group(1) : ", matchObj.group("sacc"))
            wer_Y.extend([float(matchObj.group("wer")) for x in range(now_length)])
            sacc_Y.extend([float(matchObj.group("sacc")) for x in range(now_length)])
            try:
                assert len(X) == len(wer_Y)
                assert len(X) == len(sacc_Y)
            except:
                print("出錯")
                exit()
            now_length = 0
        else:
            pass
        line_num += 1
X = X[:-11]
Y = Y[:-11]
try:
    print(len(X))
    print(len(wer_Y))
    print(len(sacc_Y))
    assert len(X) == len(wer_Y)
    assert len(X) == len(sacc_Y)
except:
    print("最后一次循环有问题")

plt.plot(X, Y, label="loss")
plt.plot(X, wer_Y, label="wer")
plt.plot(X, sacc_Y, label="sacc")

plt.legend(loc="upper right")

plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(OUTPUT_FILE)
# 在ipython的交互环境中需要这句话才能显示出来
plt.show()
