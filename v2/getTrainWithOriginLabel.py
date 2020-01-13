# coding:utf-8
from xml.dom.minidom import parse
import xml.dom.minidom
import os
import cv2
import numpy as np
import traceback
import math
from utils import BoxDiagram

'''
作者的caption使用了统一符号的操作，getTrain.py未做相应处理，现将其
原caption文件用作标注文件，并生成.align文件
'''

FLAG = "train"

# ************************加载符号词表*************************************
file = open("dictionary.txt")
symbol_dict = {}
max_length = -1  # 最长的symbol所包含的字符长度
max_length_symbol = ""
while 1:
    lines = file.readlines(100000)
    if not lines:
        break
    for line in lines:
        temp_list = line.split()
        if len(temp_list) == 2:
            a = temp_list[0].strip()
            b = temp_list[1].strip()
            symbol_dict[a] = b
            if len(a) > max_length:
                max_length = len(a)
                max_length_symbol = a

print(symbol_dict)
print("max_length: " + str(max_length))
print(max_length_symbol)
# *************************************************************
# 该代码使用的caption文件为作者原本caption，其做了格式统一化处理，先加载其文件
origin_dict = {}
origin_train_caption = open("caption/origin_" + FLAG + "_caption.txt", "r")
while 1:
    lines = origin_train_caption.readlines(100000)
    if not lines:
        break
    for line in lines:
        temp_list = line.strip().split()
        temp_str = ""
        for i, x in enumerate(temp_list[1:]):
            if x != "":
                temp_str += x
            if i != len(temp_list[1:]) - 1:
                temp_str += " "
        origin_dict[temp_list[0]] = temp_str
origin_train_caption.close()

# ***********************************************************************

# 因为CROHME 2014 数据不同类型文件坐标不统一，因此在去除重复点之前先对齐到标准文件
# standard_X standard_Y则是标准，根据统一的641个文件计算而出
bias = 1
standard_X = 659.0263006313946 * bias

total = 0
annotation_type = {}
error_count = 0
num_remove_point = 0

'''
scale为缩放比例，对于某些以浮点数为坐标的inkml，cv2无法画出它们的坐标，需要线性对标到标准文件
在删除重复点时选用的对标文件为106_Nina_origin.jpg系列
'''


def scaleTrace(now_x, traceid2xy):
    max_x = -1
    max_y = -1
    min_x = 99999999999
    min_y = 99999999999
    new_traceid2xy = []
    scale_x = now_x / standard_X
    for i, x in enumerate(traceid2xy):
        temp_list = []
        for j, y in enumerate(x):
            temp_list.append([y[0] / scale_x, y[1] / scale_x, y[2], y[3]])
            if y[0] / scale_x > max_x:
                max_x = y[0] / scale_x
            if y[1] / scale_x > max_y:
                max_y = y[1] / scale_x
            if y[0] / scale_x < min_x:
                min_x = y[0] / scale_x
            if y[1] / scale_x < min_y:
                min_y = y[1] / scale_x
        new_traceid2xy.append(temp_list)
    return new_traceid2xy, max_x, max_y, min_x, min_y


'''
按照trace把公式图片画出来
'''


def drawPictureByTrace(traceid2xy, filename, pic_output_path, min_x, min_y, max_x, max_y):
    img_ = np.full((int(max_y - min_y) + 1, int(max_x - min_x) + 1, 3), (255, 255, 255), np.uint8)
    for i, x in enumerate(traceid2xy):
        for j, y in enumerate(x):
            if j == 0:
                continue
            try:
                cv2.line(img_, (int(x[j - 1][0] - min_x), int(x[j - 1][1] - min_y)),
                         (int(y[0] - min_x), int(y[1] - min_y)), (0, 0, 0), 2)
            except:
                traceback.print_exc()
                print("what")
    cv2.imwrite(pic_output_path + "/" + filename + ".jpg", img_)


'''
传入symbol_label，返回以空格隔开的序列
'''


def deal_symbol_label(symbol_label):
    result_str = ""
    start_index = 0
    while start_index < len(symbol_label):
        if symbol_label[start_index] == " " or symbol_label[start_index] == "$":
            start_index += 1
            continue
        sign = 0
        i = 0
        while i < max_length:
            if symbol_label[start_index:start_index + max_length - i] in symbol_dict:
                result_str += symbol_label[start_index:start_index + max_length - i]
                result_str += " "
                start_index = start_index + max_length - i
                sign = 1
                break;
            i += 1
        if sign == 0:
            print(symbol_label + " 中有字典中没有的字符")
            start_index = len(symbol_label)
            result_str = ""
    return result_str


'''
计算T_cos与T_dis，去除多余点，以每个笔画为单位
'''


def rve_duplicate(traceid2xy, T_dis, T_cos):
    count = 0
    for i, x in enumerate(traceid2xy):
        j = 0
        while j < len(x):
            if j == 0:
                # temp_list.append([x[j][0], x[j][1], x[j][2], x[j][3]])
                j += 1
                continue
            real_dis = ((x[j][0] - x[j - 1][0]) ** 2 + (x[j][1] - x[j - 1][1]) ** 2) ** 0.5
            if not real_dis < T_dis:
                # temp_list.append([x[j][0], x[j][1], x[j][2], x[j][3]])
                j += 1
            else:
                if j != len(x) - 1:
                    x.pop(j)
                    print("因为距离删除一个点")
                else:
                    print("原本要删除的点为抬笔点，保留")
                    j += 1

                count += 1
    for i, x in enumerate(traceid2xy):
        j = 0
        while j < len(x):
            if j == 0 or j == len(x) - 1:
                j += 1
                continue
            if (((x[j][0] - x[j - 1][0]) ** 2 + (x[j][1] - x[j - 1][1]) ** 2) ** 0.5 * (
                    ((x[j + 1][0] - x[j][0]) ** 2 + (x[j + 1][1] - x[j][1]) ** 2) ** 0.5)) == 0:
                j += 1
                continue
            real_cos = abs(
                ((x[j][0] - x[j - 1][0]) * (x[j + 1][0] - x[j][0]) + (x[j][1] - x[j - 1][1]) * (
                        x[j + 1][1] - x[j][1])) / (
                        ((x[j][0] - x[j - 1][0]) ** 2 + (x[j][1] - x[j - 1][1]) ** 2) ** 0.5 * (
                        ((x[j + 1][0] - x[j][0]) ** 2 + (x[j + 1][1] - x[j][1]) ** 2) ** 0.5)))
            if not real_cos < T_cos:
                j += 1
            else:
                x.pop(j)
                print("因为角度删除一个点")
                count += 1
    return traceid2xy, count


'''
抽取8维特征，前6维分别为x，y，xi+1 - xi，yi+1 - yi，xi+2 - xi，yi+2 - yi
最后两维为 1 0 代表落笔 0 1 代表提笔 
'''


def feature_extraction(traceid2xy):
    new_traceid2xy = []
    for i, x in enumerate(traceid2xy):
        if i != len(traceid2xy) - 1:
            if i != len(traceid2xy) - 2:
                new_traceid2xy.append([x[0], x[1], traceid2xy[i + 1][0] - x[0], traceid2xy[i + 1][1] - x[1],
                                       traceid2xy[i + 2][0] - x[0], traceid2xy[i + 2][1] - x[1], 0.0, x[2], x[3]])
            else:
                new_traceid2xy.append([x[0], x[1], traceid2xy[i + 1][0] - x[0], traceid2xy[i + 1][1] - x[1],
                                       0.0, 0.0, 0.0, x[2], x[3]])
        else:
            new_traceid2xy.append([x[0], x[1], 0.0, 0.0, 0.0, 0.0, 0.0, x[2], x[3]])
    return new_traceid2xy


'''
减均值μx 除以标准差δx

'''


def z_score(traceid2xy):
    u_x_numerator = 0
    u_x_denominator = 0
    u_y_numerator = 0
    u_y_denominator = 0
    for i, x in enumerate(traceid2xy):
        for j, y in enumerate(x):
            if j == 0:
                continue
            L = ((y[0] - x[j - 1][0]) ** 2 + (y[1] - x[j - 1][1]) ** 2) ** 0.5
            u_x_numerator += L * (y[0] + x[j - 1][0]) / 2
            u_x_denominator += L
            u_y_numerator += L * (y[1] + x[j - 1][1]) / 2
            u_y_denominator += L
    u_x = u_x_numerator / u_x_denominator
    u_y = u_y_numerator / u_y_denominator
    delta_x_numerator = 0
    delta_x_denominator = 0
    for i, x in enumerate(traceid2xy):
        for j, y in enumerate(x):
            if j == 0:
                continue
            L = ((y[0] - x[j - 1][0]) ** 2 + (y[1] - x[j - 1][1]) ** 2) ** 0.5
            delta_x_numerator += L / 3 * (
                    (y[0] - u_x) ** 2 + (x[j - 1][0] - u_x) ** 2 + (x[j - 1][0] - u_x) * (y[0] - u_x))
            delta_x_denominator += L

    delta_x = (delta_x_numerator / delta_x_denominator) ** 0.5

    new_traceid2xy = []
    for i, x in enumerate(traceid2xy):
        for j, y in enumerate(x):
            new_traceid2xy.append([(y[0] - u_x) / delta_x, (y[1] - u_y) / delta_x, y[2], y[3]])
    return new_traceid2xy


'''
将原caption文件结果以空格隔开形式返回
'''


def deal_single_file(parent, filename, caption_file, ascii_output_path, align_output_path, pic_output_path,
                     origin_dict):
    file_name = os.path.join(parent, filename)
    symbol_label = ""
    document = xml.dom.minidom.parse(file_name + ".inkml")
    collection = document.documentElement
    count = 0
    # movie.getElementsByTagName('description')[0]
    annotations = collection.getElementsByTagName("annotation")
    # 获取 symbol_label
    for annotation in annotations:
        if annotation.hasAttribute("type"):
            if annotation.getAttribute("type") == "truth":
                count += 1
                if count == 1:
                    symbol_label = annotation.childNodes[0].data
                if count == 2:
                    temp = annotation.childNodes[0].data.strip()
                    if not temp in annotation_type:
                        annotation_type[temp] = 1
                        print("新类型：" + file_name + ".inkml")
                    else:
                        annotation_type[temp] += 1
                    break;

    assert symbol_label != ""

    # **************************原代码**************************
    # result = deal_symbol_label(symbol_label.strip())
    # ** ** ** ** ** ** ** ** **新代码** ** ** **
    '''
    将原caption文件结果以空格隔开形式返回
    '''
    if filename in origin_dict:
        result = origin_dict[filename]
    else:
        result = ""

    # ***********************************************************
    traceid2xy = []
    # 将所有点提出
    traces = collection.getElementsByTagName("trace")

    total_x = 0
    total_y = 0
    num_points = 0
    for trace in traces:
        temp_result = []
        trace_str = trace.childNodes[0].data.strip()
        temp_list = trace_str.split(",")
        for i, xy in enumerate(temp_list):
            x_y = xy.split()
            x = float(x_y[0])
            y = float(x_y[1])
            total_x += x
            total_y += y
            if i != len(temp_list) - 1:
                temp_result.append([x, y, 1, 0])
            else:
                temp_result.append([x, y, 0, 1])
            num_points += 1
        traceid2xy.append(temp_result)

    avg_x = total_x / float(num_points)
    avg_y = total_y / float(num_points)

    print("avg_x is " + str(avg_x))
    print("avg_y is " + str(avg_y))

    # ***********************计算标准文件用***********************
    # return avg_x, avg_y
    # ************************************************************

    # 进行scale变换
    traceid2xy, max_x, max_y, min_x, min_y = scaleTrace(avg_x, traceid2xy)

    print("移除重复点之前")
    drawPictureByTrace(traceid2xy, filename + "_origin", pic_output_path, min_x, min_y, max_x, max_y)
    # 移除重复点
    # traceid2xy, num_remove_point = rve_duplicate(traceid2xy, 0.5, -9999)
    # traceid2xy, num_remove_point = rve_duplicate(traceid2xy, .5 * bias, math.pi / 8)
    traceid2xy, num_remove_point = rve_duplicate(traceid2xy, 1.0 * bias, math.pi / 4)
    print("移除重复点之后")
    drawPictureByTrace(traceid2xy, filename + "_after", pic_output_path, min_x, min_y, max_x, max_y)
    # Z-score正则化,二元列表的行元素不再为stroken，为points
    traceid2xy = z_score(traceid2xy)
    # 使用箱线图去除异常点
    traceid2xy = BoxDiagram(traceid2xy)
    # 转换为8维特征
    traceid2xy = feature_extraction(traceid2xy)

    # ***************将feature输出到文件*******************
    o = open(os.path.join(ascii_output_path, filename) + ".ascii", "w")
    for x in traceid2xy:
        for j, y in enumerate(x):
            o.write(str(y))
            if j != len(x) - 1:
                o.write(" ")
        o.write("\n")
    o.close()
    # ***************将symbol对应的trace_id提出*******************
    symbol2traceId = {}
    traceGroups = collection.getElementsByTagName("traceGroup")
    for i, x in enumerate(traceGroups):
        if i == 0:
            continue
        annotation = x.getElementsByTagName('annotation')[0]
        symbol2traceId[annotation.childNodes[0].data.strip()] = []
        for y in x.getElementsByTagName('traceView'):
            symbol2traceId[annotation.childNodes[0].data.strip()].append(int(y.getAttribute("traceDataRef").strip()))

    print(symbol2traceId)

    o = open(os.path.join(align_output_path, filename) + ".align", "w")

    iteration = result.strip().split()
    for j, x in enumerate(iteration):
        o.write(x + " ")
        if x in symbol2traceId:
            for i, y in enumerate(symbol2traceId[x]):
                if i != len(symbol2traceId[x]) - 1:
                    o.write(str(y) + " ")
                else:
                    o.write(str(y))
        else:
            o.write("-1")
        if j != len(iteration) - 1:
            o.write("\n")
    o.close()
    # ***********************************************************
    if result != "":
        caption_file.write(filename)
        caption_file.write("\t")
        caption_file.write(result)
        caption_file.write("\n")
        return True, num_remove_point
    else:
        return False, num_remove_point


# *********************计算标准文件所用***********************
total_avg_x = 0
total_avg_y = 0
# ************************************************************

input_path = FLAG + "_data/"
# input_path = "another_" + FLAG + "_data/"
caption_output_path = "./caption/"
ascii_output_path = "./on-ascii-" + FLAG + "/"
align_output_path = "./on-align-" + FLAG + "/"
pic_output_path = "./pic-" + FLAG + "/"

if not os.path.exists(caption_output_path):
    os.makedirs(caption_output_path)
if not os.path.exists(ascii_output_path):
    os.makedirs(ascii_output_path)
if not os.path.exists(align_output_path):
    os.makedirs(align_output_path)
if not os.path.exists(pic_output_path):
    os.makedirs(pic_output_path)

caption_file = caption_output_path + FLAG + "_caption.txt"
caption_file = open(caption_file, "w")
for parent, dirnames, filenames in os.walk(input_path, followlinks=True):
    for filename in filenames:
        file_path = os.path.join(parent, filename)
        print('文件名：%s' % filename)
        print('文件完整路径：%s\n' % file_path)
        if file_path[-6:] == ".inkml":
            sign, _ = deal_single_file(parent, filename[:-6], caption_file, ascii_output_path, align_output_path,
                                       pic_output_path, origin_dict)
            if False == sign:
                error_count += 1
            num_remove_point += _
            # *********************计算标准文件所用***********************
            # delta_x, delta_y = deal_single_file(parent, filename[:-6], caption_file, ascii_output_path,
            #                                     align_output_path,
            #                                     pic_output_path)
            # total_avg_x += delta_x
            # total_avg_y += delta_y
            # ************************************************************
            total += 1

# *********************计算标准文件所用***********************
# print("标准X " + str(total_avg_x / total))
# print("标准Y " + str(total_avg_y / total))
# ************************************************************

caption_file.close()
print("annotation type:")
for x, y in annotation_type.items():
    print(x + ":" + str(y))

print("共处理文件" + str(total) + "个,错误文件数" + str(error_count))
if total - error_count == 0:
    print("全错")
else:
    print("平均每个文件移除点个数：" + str(float(num_remove_point) / (total - error_count)))
