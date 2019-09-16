# coding:utf-8
from xml.dom.minidom import parse
import xml.dom.minidom
import os

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

# ***********************************************************************

total = 0
annotation_type = {}
error_count = 0

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
    new_traceid2xy = []
    for i, x in enumerate(traceid2xy):
        temp_list = []
        for j, y in enumerate(x):
            if j == 0:
                temp_list.append([y[0], y[1], y[2], y[3]])
                continue
            real_dis = ((y[0] - x[j - 1][0]) ** 2 + (y[1] - x[j - 1][1]) ** 2) ** 0.5
            # print(real_dis)
            if not real_dis < T_dis:
                temp_list.append([y[0], y[1], y[2], y[3]])
        new_traceid2xy.append(temp_list)
    traceid2xy = []
    for i, x in enumerate(new_traceid2xy):
        temp_list = []
        for j, y in enumerate(x):
            if j == 0 or j == len(x) - 1:
                temp_list.append([y[0], y[1], y[2], y[3]])
                continue
            if (((y[0] - x[j - 1][0]) ** 2 + (y[1] - x[j - 1][1]) ** 2) ** 0.5 * (
                    ((x[j + 1][0] - y[0]) ** 2 + (x[j + 1][1] - y[1]) ** 2) ** 0.5)) == 0:
                continue

            real_cos = ((y[0] - x[j - 1][0]) * (x[j + 1][0] - y[0]) + (y[1] - x[j - 1][1]) * (x[j + 1][1] - y[1])) / (
                    ((y[0] - x[j - 1][0]) ** 2 + (y[1] - x[j - 1][1]) ** 2) ** 0.5 * (
                    ((x[j + 1][0] - y[0]) ** 2 + (x[j + 1][1] - y[1]) ** 2) ** 0.5))
            if not real_cos < T_cos:
                temp_list.append([y[0], y[1], y[2], y[3]])
        traceid2xy.append(temp_list)
    return traceid2xy


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


def deal_single_file(parent, filename, caption_file, ascii_output_path, align_output_path):
    file_name = os.path.join(parent, filename)
    symbol_label = ""
    document = xml.dom.minidom.parse(file_name + ".inkml")
    collection = document.documentElement
    count = 0
    # movie.getElementsByTagName('description')[0]
    annotations = collection.getElementsByTagName("annotation")
    for annotation in annotations:
        if annotation.hasAttribute("type"):
            if annotation.getAttribute("type") == "truth":
                count += 1
                if count == 1:
                    symbol_label = annotation.childNodes[0].data
                    # print(symbol_label)
                if count == 2:
                    temp = annotation.childNodes[0].data.strip()
                    if not temp in annotation_type:
                        annotation_type[temp] = 1
                        print("新类型：" + file_name + ".inkml")
                    else:
                        annotation_type[temp] += 1
                    break;

    assert symbol_label != ""
    result = deal_symbol_label(symbol_label.strip())

    traceid2xy = []
    # 将所有点提出
    traces = collection.getElementsByTagName("trace")

    for trace in traces:
        temp_result = []
        trace_str = trace.childNodes[0].data.strip()
        temp_list = trace_str.split(",")
        for i, xy in enumerate(temp_list):
            x_y = xy.split()
            x = x_y[0]
            y = x_y[1]
            if i != len(temp_list) - 1:
                temp_result.append([float(x), float(y), 1, 0])
            else:
                temp_result.append([float(x), float(y), 0, 1])
            # print("x is " + x, "y is " + y)

        traceid2xy.append(temp_result)
    # print(traceid2xy)

    # 移除重复点
    traceid2xy = rve_duplicate(traceid2xy, -100, -999)
    # Z-score正则化,二元列表的行元素不再为stroken，为points
    traceid2xy = z_score(traceid2xy)
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
    for x in iteration:
        o.write(x + " ")
        if x in symbol2traceId:
            for i, y in enumerate(symbol2traceId[x]):
                if i != len(symbol2traceId[x]) - 1:
                    o.write(str(y) + " ")
                else:
                    o.write(str(y))
        else:
            o.write("-1")
        o.write("\n")
    o.close()
    # ***********************************************************
    if result != "":
        caption_file.write(filename)
        caption_file.write("\t")
        caption_file.write(result)
        caption_file.write("\n")
        return True
    else:
        return False


input_path = "data/"
caption_output_path = "./caption/"
ascii_output_path = "./on-ascii-" + FLAG + "/"
align_output_path = "./on-align-train/"
if not os.path.exists(caption_output_path):
    os.makedirs(caption_output_path)
if not os.path.exists(ascii_output_path):
    os.makedirs(ascii_output_path)
if not os.path.exists(align_output_path):
    os.makedirs(align_output_path)
caption_file = caption_output_path + FLAG + "_caption.txt"
caption_file = open(caption_file, "w")
for parent, dirnames, filenames in os.walk(input_path, followlinks=True):
    for filename in filenames:
        file_path = os.path.join(parent, filename)
        print('文件名：%s' % filename)
        print('文件完整路径：%s\n' % file_path)
        if file_path[-6:] == ".inkml":
            if False == deal_single_file(parent, filename[:-6], caption_file, ascii_output_path, align_output_path):
                error_count += 1
            total += 1

caption_file.close()
print("annotation type:")
for x, y in annotation_type.items():
    print(x + ":" + str(y))
print("共处理文件" + str(total) + "个,错误文件数" + str(error_count))
