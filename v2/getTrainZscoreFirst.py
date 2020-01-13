# coding:utf-8
from xml.dom.minidom import parse
import xml.dom.minidom
import os
import cv2
import numpy as np
import traceback
import math
from utils import BoxDiagram, z_score_stroken_level, my_z_score, z_score, zscore_first_feature_extraction

'''
作者的caption使用了统一符号的操作，getTrain.py未做相应处理，现将其
原caption文件用作标注文件，并生成.align文件

该版本将z-score放在之前操作，之后再z-score基础上进行统一去重
'''

FLAG = "test"

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
origin_train_caption = open("../TAP_DATA/caption/origin_" + FLAG + "_caption.txt", "r")
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

from utils import standard_X

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
将原caption文件结果以空格隔开形式返回
'''


def deal_single_file(parent, filename, caption_file, ascii_output_path, align_output_path, pic_output_path,
                     origin_dict, pre_u_x,
                     pre_delta_x,
                     pre_u_y):
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
    # traceid2xy, max_x, max_y, min_x, min_y = scaleTrace(avg_x, traceid2xy)

    # Z-score正则化,二元列表的行元素不再为stroken，为points

    # traceid2xy, pre_u_x, pre_delta_x, pre_u_y = z_score_stroken_level(traceid2xy, pre_u_x,
    #                                                                   pre_delta_x,
    #                                                                   pre_u_y)
    traceid2xy = z_score(traceid2xy)
    print("移除重复点之前")
    print_traceid2xy, max_x, max_y, min_x, min_y = scaleTrace(avg_x, traceid2xy)
    drawPictureByTrace(print_traceid2xy, filename + "_origin", pic_output_path, min_x, min_y, max_x, max_y)
    # 移除重复点
    # traceid2xy, num_remove_point = rve_duplicate(traceid2xy, 0.5, -9999)
    # traceid2xy, num_remove_point = rve_duplicate(traceid2xy, .5 * bias, math.pi / 8)
    # traceid2xy, num_remove_point = rve_duplicate(traceid2xy, 0.25 * bias, math.pi / 12)

    traceid2xy, num_remove_point = rve_duplicate(traceid2xy, .005, -999999)

    print("移除重复点之后")
    print_traceid2xy, max_x, max_y, min_x, min_y = scaleTrace(avg_x, traceid2xy)
    drawPictureByTrace(print_traceid2xy, filename + "_after", pic_output_path, min_x, min_y, max_x, max_y)

    # 使用箱线图去除异常点
    # traceid2xy, _ = BoxDiagram(traceid2xy)
    # num_remove_point += _
    # 转换为8维特征
    traceid2xy = zscore_first_feature_extraction(traceid2xy)

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
    # ***************symbol有可能会重复，需要duplication_dict辅助**********
    symbol2traceId = {}
    duplication_dict = {}
    traceGroups = collection.getElementsByTagName("traceGroup")
    for i, x in enumerate(traceGroups):
        if i == 0:
            continue
        annotation = x.getElementsByTagName('annotation')[0]
        temp_key = annotation.childNodes[0].data.strip()

        # 特殊字符处理，形如\frac般的字符，在align标注时会以-来标注，与负号混淆，需要单独处理
        if temp_key == r"-":

            mfrac_list = {}
            for lp in collection.getElementsByTagName("mfrac"):
                mfrac_list[lp.getAttribute("xml:id").strip()] = 1
            annotationXML = x.getElementsByTagName('annotationXML')
            if len(annotationXML) != 0:
                if annotationXML[0].getAttribute("href").strip() in mfrac_list:
                    temp_key = r"\frac"

        if temp_key in symbol2traceId:
            if temp_key in duplication_dict:
                duplication_dict[temp_key] += 1
            else:
                duplication_dict[temp_key] = 1
            symbol2traceId[temp_key + "_" + str(duplication_dict[temp_key])] = []
            for y in x.getElementsByTagName('traceView'):
                symbol2traceId[temp_key + "_" + str(duplication_dict[temp_key])].append(
                    int(y.getAttribute("traceDataRef").strip()))
        else:
            symbol2traceId[temp_key] = []
            for y in x.getElementsByTagName('traceView'):
                symbol2traceId[temp_key].append(
                    int(y.getAttribute("traceDataRef").strip()))

    # print(symbol2traceId)

    # 用来记录重复元素已输出多少次的字典
    duplication_count_dict = {}

    o = open(os.path.join(align_output_path, filename) + ".align", "w")

    iteration = result.strip().split()
    for j, x in enumerate(iteration):
        o.write(x + " ")

        if x == r"<":
            x = r"\lt"
        if x == r"\cdots":
            x = r"\ldots"
        if x in symbol2traceId:
            for i, y in enumerate(symbol2traceId[x]):
                if i != len(symbol2traceId[x]) - 1:
                    o.write(str(y) + " ")
                else:
                    o.write(str(y))
            symbol2traceId.pop(x)
        else:
            if x in duplication_dict:
                if x not in duplication_count_dict:
                    duplication_count_dict[x] = 1
                    for i, y in enumerate(symbol2traceId[x + "_1"]):
                        if i != len(symbol2traceId[x + "_1"]) - 1:
                            o.write(str(y) + " ")
                        else:
                            o.write(str(y))
                    symbol2traceId.pop(x + "_1")
                else:
                    duplication_count_dict[x] += 1
                    n = duplication_count_dict[x]
                    try:
                        for i, y in enumerate(symbol2traceId[x + "_" + str(n)]):
                            if i != len(symbol2traceId[x + "_" + str(n)]) - 1:
                                o.write(str(y) + " ")
                            else:
                                o.write(str(y))
                        symbol2traceId.pop(x + "_" + str(n))
                    except:
                        # 该地方报错针对的是formulaire030-equation051这类文件出现n_n写作n造成的错误
                        # 作者的处理是什么也不输出，我们也不输出
                        pass
            else:

                o.write("-1")
        if j != len(iteration) - 1:
            o.write("\n")
    o.close()
    if len(symbol2traceId) != 0:
        print(symbol2traceId)
    # 输出完align文件，symbol2traceId一定是空表
    try:
        assert len(symbol2traceId) == 0
    except:
        os.remove(os.path.join(align_output_path, filename) + ".align")
        os.remove(os.path.join(ascii_output_path, filename) + ".ascii")
        return False, num_remove_point, pre_u_x, pre_delta_x, pre_u_y

    # ***********************************************************
    if result != "":
        caption_file.write(filename)
        caption_file.write("\t")
        caption_file.write(result)
        caption_file.write("\n")
        return True, num_remove_point, pre_u_x, pre_delta_x, pre_u_y
    else:
        return False, num_remove_point, pre_u_x, pre_delta_x, pre_u_y


# *********************计算标准文件所用***********************
total_avg_x = 0
total_avg_y = 0
# ************************************************************

input_path = "../TAP_DATA/" + FLAG + "_data/"
# input_path = "another_" + FLAG + "_data/"

# input_path = "../TAP_DATA/test_fold"

caption_output_path = "../TAP_DATA/" + "caption/"
ascii_output_path = "../TAP_DATA/" + "on-ascii-" + FLAG + "/"
align_output_path = "../TAP_DATA/" + "on-align-" + FLAG + "/"
pic_output_path = "../TAP_DATA/" + "pic-" + FLAG + "/"

pre_u_x = 0
pre_delta_x = 0
pre_u_y = 0

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
            sign, _, pre_u_x, pre_delta_x, pre_u_y = deal_single_file(parent, filename[:-6], caption_file,
                                                                      ascii_output_path, align_output_path,
                                                                      pic_output_path, origin_dict, pre_u_x,
                                                                      pre_delta_x,
                                                                      pre_u_y)
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
