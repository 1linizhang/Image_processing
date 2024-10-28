import cv2
import numpy as np

inputf = '7e7ca4ac00ddf608798e4ede18839bc8(1).mp4'
inputb = 'f34a84dfaef8377ff47751f4fb917361.mp4'
output = 'output1.mp4'

# 打开视频文件
f_cap = cv2.VideoCapture(inputf)
b_cap = cv2.VideoCapture(inputb)

# 获取视频的宽、高和帧率
fwidth = int(f_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fheight = int(f_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ffps = round(f_cap.get(cv2.CAP_PROP_FPS))

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
capWrite = cv2.VideoWriter(output, fourcc, ffps, (fwidth, fheight))

# 获取背景视频的帧率和总帧数
bfps = round(b_cap.get(cv2.CAP_PROP_FPS))
bframeCount = int(b_cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 边缘检测函数
def create_edge_mask(binary):
    img = np.zeros_like(binary, dtype=np.uint8)
    img[binary == 0] = 255  # 反转黑白，边缘为白色
    return img

while f_cap.isOpened():
    retf, framef = f_cap.read()
    if not retf:
        break

    # 计算当前帧索引
    cindex = int(f_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    bindex = int((cindex * bfps / ffps) % bframeCount)
    b_cap.set(cv2.CAP_PROP_POS_FRAMES, bindex)

    retb, frameb = b_cap.read()
    if not retb:
        b_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        retb, frameb = b_cap.read()

    # 转换为灰度图并进行二值化
    fgray = cv2.cvtColor(framef, cv2.COLOR_BGR2GRAY)
    _, binary_fframe = cv2.threshold(fgray, 128, 255, cv2.THRESH_BINARY)
    edge_mask = create_edge_mask(binary_fframe)

    # 确保背景帧与前景帧尺寸一致
    if frameb.shape[:2] != framef.shape[:2]:
        frameb = cv2.resize(frameb, (framef.shape[1], framef.shape[0]))

    # 提取前景（边缘内的颜色）
    fore = cv2.bitwise_and(framef, framef, mask=edge_mask)
    # 提取背景（边缘外的颜色）
    back_mask = cv2.bitwise_not(edge_mask)
    back = cv2.bitwise_and(frameb, frameb, mask=back_mask)

    # 合成前景和背景
    combined_frame = cv2.add(fore, back)

    # 保存调试图像
    cv2.imwrite("1.png", fore)
    cv2.imwrite("11.png", back)

    capWrite.write(combined_frame)

# 释放视频资源
f_cap.release()
b_cap.release()
capWrite.release()
cv2.destroyAllWindows()