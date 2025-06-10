#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
图像分割 - 遥感图像数据预处理模块

功能：加载 TM 影像并执行辐射校正、几何校正及图像增强处理

输入：原始 TM 影像文件路径  
输出：校正并增强后的多波段 GeoTIFF 影像  
'''

import numpy as np
from osgeo import gdal
import cv2




def load_tm_image(file_path):
    """
    加载TM影像数据
    
    参数:
        file_path: TM影像文件路径
    
    返回:
        bands_data: 包含所有波段数据的列表
        geotransform: 地理变换信息
        projection: 投影信息
    """
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise Exception("无法打开文件: " + file_path)

    # 获取图像基本信息
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands_count = dataset.RasterCount

    # 获取地理参考信息
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    # 读取所有波段数据
    bands_data = []
    for i in range(1, bands_count + 1):
        band = dataset.GetRasterBand(i)
        data = band.ReadAsArray()
        bands_data.append(data)

    print(f"成功加载影像, 尺寸: {width}x{height}, 波段数: {bands_count}")
    return bands_data, geotransform, projection

def radiometric_calibration(bands_data):
    """
    对TM影像进行辐射定标
    
    参数:
        bands_data: 包含所有波段数据的列表
    
    返回:
        calibrated_bands: 辐射校正后的波段数据
    """
    # TM影像的辐射定标参数(示例值，实际应根据具体数据调整)
    gain = [0.671339, 1.322205, 1.043976, 0.876024, 0.120354, 0.055376, 0.065551]
    bias = [-2.19, -4.16, -2.21, -2.39, -0.49, 1.18, -0.22]

    calibrated_bands = []
    for i, band_data in enumerate(bands_data):
        # 将DN值转换为辐射亮度
        radiance = gain[i] * band_data + bias[i]
        calibrated_bands.append(radiance)

    return calibrated_bands

def geometric_correction(bands_data, gcps):
    """
    对TM影像进行几何校正
    
    参数:
        bands_data: 包含所有波段数据的列表
        gcps: 地面控制点
    
    返回:
        corrected_bands: 几何校正后的波段数据
    """
    # 此处简化处理，实际几何校正需要GCPs和插值方法
    # 示例中仅做简单的仿射变换
    corrected_bands = []

    for band_data in bands_data:
        height, width = band_data.shape
        # 使用OpenCV的仿射变换作为简化示例
        # 实际应用中应基于GCPs计算变换矩阵
        transform_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        corrected_band = cv2.warpAffine(band_data, transform_matrix, (width, height))
        corrected_bands.append(corrected_band)

    return corrected_bands

def image_enhancement(bands_data):
    """
    图像增强处理
    
    参数:
        bands_data: 包含所有波段数据的列表
    
    返回:
        enhanced_bands: 增强后的波段数据
    """
    enhanced_bands = []

    for band_data in bands_data:
        # 线性拉伸增强
        min_val = np.min(band_data)
        max_val = np.max(band_data)
        enhanced_band = (band_data - min_val) * 255.0 / (max_val - min_val)
        enhanced_band = enhanced_band.astype(np.uint8)

        # 直方图均衡化(可选)
        # enhanced_band = cv2.equalizeHist(enhanced_band)

        enhanced_bands.append(enhanced_band)

    return enhanced_bands

def save_processed_image(bands_data, geotransform, projection, output_path):
    """
    保存处理后的影像
    
    参数:
        bands_data: 包含所有波段数据的列表
        geotransform: 地理变换信息
        projection: 投影信息
        output_path: 输出文件路径
    """
    driver = gdal.GetDriverByName("GTiff")
    bands_count = len(bands_data)

    # 获取第一个波段的形状作为输出图像的尺寸
    height, width = bands_data[0].shape

    # 创建输出数据集
    dataset = driver.Create(output_path, width, height, bands_count, gdal.GDT_Float32)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)

    # 写入波段数据
    for i, band_data in enumerate(bands_data):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(band_data)

    # 关闭数据集，刷新缓冲区
    dataset = None
    print(f"已保存处理后的影像到: {output_path}")