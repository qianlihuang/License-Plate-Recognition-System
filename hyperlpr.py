import os
import cv2
import hyperlpr3 as lpr3
import numpy as np

# 统计正确识别的数量
correct_num = 0
# 统计总图片数
total_num = 0


# 遍历图片目录
for filename in os.listdir("./images"):
    # 实例化识别对象
    catcher = lpr3.LicensePlateCatcher()
    # 读取图片
    image_path = os.path.join("./images", filename)
    image = cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),flags=1)




    # 识别结果
    result = catcher(image)
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
    total_num += 1

# 计算识别准确率
accuracy = correct_num / total_num
print("识别准确率为：{:.2f}%".format(accuracy * 100))
