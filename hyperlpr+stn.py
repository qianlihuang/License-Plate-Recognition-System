import os
import cv2
import hyperlpr3 as lpr3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.st import STNet

# 加载模型
checkpoint = torch.load('st.pt', map_location=torch.device('cpu'))
st_state_dict = checkpoint['st']

st_model = STNet()
st_model.load_state_dict(st_state_dict)
st_model.eval()


def convert_image(inp):
    # 将PyTorch张量转换为numpy图像
    inp = inp.squeeze(0).cpu()  # 去除batch维度，将图像拉平为三维数组
    inp = inp.detach().numpy().transpose((1, 2, 0))  # 转置数组，将通道维度置为最后一个
    inp = 127.5 + inp/0.0078125  # 将像素值缩放回[0, 255]范围
    inp = inp.astype('uint8')  # 转换数据类型为8位无符号整数

    return inp

# 统计正确识别的数量
correct_num = 0
correct_num_letter = 0
# 统计总图片数
total_num = 0
total_num_letter = 0

# 遍历图片目录
for filename in os.listdir("./images+"):
    # 实例化识别对象
    catcher = lpr3.LicensePlateCatcher()
    # 读取图片
    image_path = os.path.join("./images+", filename)
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), flags=1)

    # stn
    image = cv2.resize(image, (94, 24), interpolation=cv2.INTER_CUBIC)
    # 将图像转换为PyTorch张量
    image = (np.transpose(np.float32(image), (2, 0, 1)) - 127.5)*0.0078125
    img_tensor = torch.from_numpy(
        image).float().unsqueeze(0).to(torch.device('cpu'))

    # 使用模型进行推理
    with torch.no_grad():
        preds = st_model(img_tensor)

    transformed_img = convert_image(preds)

    # 识别结果
    result = catcher(transformed_img)
    if result:
        print ("识别结果:"+result[0][0])
    else:
        print ("识别结果:None")

    # 获取文件名（即车牌）
    plate = os.path.splitext(filename)[0]
    if "-2" in plate:
        plate = plate.replace("-2", "")
    print ("正确结果:"+plate)

    # 对比识别结果和文件名
    if result:
        if result[0][0] == plate:
            correct_num += 1
            correct_num_letter += 7
        else:
            # 对比每个字符
            for i in range(7):
                if result[0][0][i] == plate[i]:
                    correct_num_letter += 1
    total_num += 1
    total_num_letter += 7

# 计算识别准确率
# accuracy = correct_num / total_num
accuracy = correct_num_letter / total_num_letter
print("逐字识别准确率为：{:.2f}%".format(accuracy * 100))
