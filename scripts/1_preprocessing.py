#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
'''
图像分割 - 遥感图像预处理模块

功能：对原始TM影像进行辐射校正、几何校正和图像增强处理

输入：原始TM影像文件
输出：预处理后的影像文件

作者：Meng Yinan
日期：2025-05-26
'''



import matplotlib.pyplot as plt
import os

from modules.features.preprocessing import *
from modules.utils.set_chinese_font import set_chinese_font

set_chinese_font()

def run_preprocessing_stage(input_file, output_file, visualization_output_dir):
    """
    数据预处理主函数
    
    参数:
        input_file (str): 输入原始影像文件路径
        output_file (str): 输出预处理后影像文件路径
        visualization_output_dir (str): 可视化结果保存目录
    
    返回:
        str: 预处理后影像文件路径
    """
    print("开始数据预处理阶段...")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(visualization_output_dir, exist_ok=True)

    # 加载数据
    bands_data, geotransform, projection = load_tm_image(input_file)

    # 辐射校正
    calibrated_bands = radiometric_calibration(bands_data)

    # 几何校正 (示例中使用空的GCPs列表)
    corrected_bands = geometric_correction(calibrated_bands, [])

    # 图像增强
    enhanced_bands = image_enhancement(corrected_bands)

    # 保存处理后的影像
    save_processed_image(enhanced_bands, geotransform, projection, output_file)

    # 显示处理结果
    plt.figure(figsize=(15, 10))

    # 显示原始图像(使用4,3,2波段合成假彩色图像)
    plt.subplot(121)
    rgb = np.stack([bands_data[3], bands_data[2], bands_data[1]], axis=-1)
    # 归一化显示
    rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
    plt.imshow(rgb)
    plt.title("原始TM影像(4,3,2波段合成)")

    # 显示处理后的图像
    plt.subplot(122)
    rgb_enhanced = np.stack([enhanced_bands[3], enhanced_bands[2], enhanced_bands[1]], axis=-1)
    # 归一化为0-1范围
    rgb_enhanced = rgb_enhanced / 255.0
    plt.imshow(rgb_enhanced)
    plt.title("预处理后的TM影像(4,3,2波段合成)")

    plt.tight_layout()
    visualization_path = os.path.join(visualization_output_dir, "preprocessing_result.png")
    plt.savefig(visualization_path, dpi=300)
    plt.show()

    print("数据预处理阶段完成")
    return output_file

if __name__ == "__main__":
    # 测试模式下的默认参数
    input_file = "../data/raw/AA.tif"
    output_file = "../data/TM_image_AA_preprocessed.png/TM_image_AA_preprocessed.tif"
    visualization_output_dir = "../data"

    run_preprocessing_stage(input_file, output_file, visualization_output_dir)

# import matplotlib.pyplot as plt
# 
# from .features.preprocessing import *
# 
# from .utils.set_chinese_font import set_chinese_font
# 
# set_chinese_font()
# 
# def main():
#     # 输入和输出文件路径
#     input_file = "../data/raw/AA.tif"  # 假设的输入文件名
#     output_file = "../TM_image_AA_preprocessed.tif"
# 
#     # 加载数据
#     bands_data, geotransform, projection = load_tm_image(input_file)
# 
#     # 辐射校正
#     calibrated_bands = radiometric_calibration(bands_data)
# 
#     # 几何校正 (示例中使用空的GCPs列表)
#     corrected_bands = geometric_correction(calibrated_bands, [])
# 
#     # 图像增强
#     enhanced_bands = image_enhancement(corrected_bands)
# 
#     # 保存处理后的影像
#     save_processed_image(enhanced_bands, geotransform, projection, output_file)
# 
#     # 显示处理结果(可选)
#     plt.figure(figsize=(15, 10))
# 
#     # 显示原始图像(使用4,3,2波段合成假彩色图像)
#     plt.subplot(121)
#     rgb = np.stack([bands_data[3], bands_data[2], bands_data[1]], axis=-1)
#     # 归一化显示
#     rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
#     plt.imshow(rgb)
#     plt.title("原始TM影像(4,3,2波段合成)")
# 
#     # 显示处理后的图像
#     plt.subplot(122)
#     rgb_enhanced = np.stack([enhanced_bands[3], enhanced_bands[2], enhanced_bands[1]], axis=-1)
#     # 归一化为0-1范围
#     rgb_enhanced = rgb_enhanced / 255.0
#     plt.imshow(rgb_enhanced)
#     plt.title("预处理后的TM影像(4,3,2波段合成)")
# 
#     plt.tight_layout()
#     plt.savefig("../data/preprocessing_result.png", dpi=300)
#     plt.show()
# 
# if __name__ == "__main__":
#     main()