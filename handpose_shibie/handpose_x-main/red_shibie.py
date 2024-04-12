import cv2
import numpy as np
import os

def extract_number(filename):
    # 从文件名中提取数字部分
    return int(''.join(filter(str.isdigit, filename)))


def red_i(img_path):
    white = []
    idx = 0
    for file in sorted(os.listdir(img_path), key=extract_number):
        if '.jpg' not in file:
            continue
        idx += 1
        path1 = img_path
        # 第二部分路径
        filename = file
        # 合并路径
        full_path = os.path.join(path1, filename)
        print(full_path)

        img = cv2.imread(full_path)
        # 在彩色图像的情况下，解码图像将以b g r顺序存储通道。
        grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 从RGB色彩空间转换到HSV色彩空间
        grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)

        # H、S、V范围一：
        # lower1 = np.array([0, 43, 46])
        # lower1 = np.array([0, 145, 46])
        lower1 = np.array([0, 190, 46])
        upper1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(grid_HSV, lower1, upper1)  # mask1 为二值图像
        res1 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask1)

        # H、S、V范围二：
        # lower2 = np.array([156, 43, 46])
        # lower2 = np.array([156, 145, 46])
        lower2 = np.array([156, 190, 46])
        upper2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(grid_HSV, lower2, upper2)
        res2 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask2)

        # 将两个二值图像结果 相加
        mask3 = mask1 + mask2
        # mask3 = mask2

        # # 结果显示
        # cv2.imshow("mask3", mask3)
        # cv2.waitKey(33)
        # cv2.destroyAllWindows()
        # print("flag")

        # 求取红色区域坐标
        height, width = mask3.shape
        # 存储白色像素的坐标
        white_pixel_coordinates = []
        # 遍历图像并检查每个像素值
        for y in range(height):
            for x in range(width):
                if mask3[y, x] == 255:
                    white_pixel_coordinates.append((x, y))
        # 打印白色像素的坐标
        # print("第%d张图的红色区域坐标：" % idx)
        # print(white_pixel_coordinates)
        # white.append(white_pixel_coordinates)
        # if idx == 1:
        #     break
    # print(white)
    return white


# red_i('./image/')

def red_j(img_path):
    white = []
    idx = 0
    for file in sorted(os.listdir(img_path), key=extract_number):
        if '.jpg' not in file:
            continue
        idx += 1
        path1 = img_path
        # 第二部分路径
        filename = file
        # 合并路径
        full_path = os.path.join(path1, filename)
        print(full_path)

        img = cv2.imread(full_path)
        # 在彩色图像的情况下，解码图像将以b g r顺序存储通道。
        grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 从RGB色彩空间转换到HSV色彩空间
        grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)

        # H、S、V范围一：
        # lower1 = np.array([0, 43, 46])
        # lower1 = np.array([0, 145, 46])
        lower1 = np.array([0, 190, 46])
        upper1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(grid_HSV, lower1, upper1)  # mask1 为二值图像
        res1 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask1)

        # H、S、V范围二：
        # lower2 = np.array([156, 43, 46])
        # lower2 = np.array([156, 145, 46])
        lower2 = np.array([156, 190, 46])
        upper2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(grid_HSV, lower2, upper2)
        res2 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask2)

        # 将两个二值图像结果 相加
        mask3 = mask1 + mask2
        # mask3 = mask2

        # # 结果显示
        # cv2.imshow("mask3", mask3)
        # cv2.waitKey(33)
        # cv2.destroyAllWindows()
        # print("flag")

        # 求取红色区域坐标
        height, width = mask3.shape
        # print(height, width)
        # ker = np.ones((4, 4))
        # 存储白色像素的坐标
        white_pixel_coordinates = []
        # 遍历图像并检查每个像素值
        nonzero_indices = np.nonzero(mask3)
        white_pixel_coordinates = list(zip(nonzero_indices[1], nonzero_indices[0]))
        white.append(white_pixel_coordinates)
        # print(white_pixel_coordinates)
        # cv2.circle(img,   (1799, 3091), 5,(0,255,255), -1)
        # cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
        # cv2.imshow('img', img)
        # cv2.waitKey(3300)
        # if idx == 1:
        #     break
    # print(white)
    return white

# red_j('./image/')