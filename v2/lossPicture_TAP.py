'''
将CROHME_TAP_%d.log转化为图片
'''

import matplotlib.pyplot as plt  # 约定俗成的写法plt
# 首先定义两个函数（正弦&余弦）
import re

INPUT_FILE = "TAP_BETTER_LABEL.log"
OUTPUT_FILE = "loss_pic_tap_better_label.png"


def calculatedRatio(sacc_Y):
    try:
        for i, x in enumerate(sacc_Y):
            if i == 0:
                continue
            if x / sacc_Y[i - 1] < 0.5 or x / sacc_Y[i - 1] > 1.5:
                return False
        return True
    except:
        return False


file = open(INPUT_FILE)
X = []
Y = []
wer_Y = []
sacc_Y = []
line_num = 0
now_length = 0
sign = 0
convergence_sacc = 0
convergence_wer = 0
convergence_loss = 0

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

            if sign == 0 and len(sacc_Y) > 5:
                if True == calculatedRatio(sacc_Y[-5:]):
                    convergence_sacc = sacc_Y[-1]
                if True == calculatedRatio(wer_Y[-5:]):
                    convergence_wer = wer_Y[-1]
                if True == calculatedRatio(Y[-5:]):
                    convergence_loss = Y[-1]
            try:
                assert len(X) == len(wer_Y)
                assert len(X) == len(sacc_Y)
            except:
                print("出错")
                exit()
            now_length = 0
        else:
            pass
        line_num += 1
X = X[:-now_length]
Y = Y[:-now_length]
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

print("收敛时sacc为" + str(convergence_sacc))
print("收敛时wer为" + str(convergence_wer))
print("收敛时loss为" + str(convergence_loss))
