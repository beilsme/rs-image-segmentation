#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
'''
图像分割 - ROI 掩膜生成模块

功能：根据交互采样点和参考影像创建像素级 ROI 掩膜

输入：  
  - 采样坐标与标签文件（PKL 格式）  
  - 参考影像文件（GeoTIFF 格式）

输出：  
  - ROI 掩膜数组（NumPy .npy 格式）

作者：Meng Yinan  
日期：2025-05-26
'''

import os
import pickle
import numpy as np
import rasterio

def generate_roi_mask_from_samples(
        samples_pkl_path: str,
        reference_image_path: str,
        roi_mask_out: str
):
    """
    从采样点生成 ROI 掩膜 (H×W 整数数组)。
    samples_pkl_path: pickle 文件，内容是 (coords, labels)
    reference_image_path: 用于获取 H, W 的 TIFF 路径
    roi_mask_out: 输出 npy 文件路径
    """
    # 1. 读取坐标和标签
    with open(samples_pkl_path, "rb") as f:
        coords, labels = pickle.load(f)
    coords = np.array(coords, dtype=int)  # (N,2): [ [x1,y1], [x2,y2], ... ]
    labels = np.array(labels, dtype=int)  # (N,)

    # 2. 读取参考影像的尺寸
    with rasterio.open(reference_image_path) as ds:
        H, W = ds.height, ds.width

    # 3. 初始化全零掩膜
    roi_mask = np.zeros((H, W), dtype=np.int16)

    # 4. 填值
    for (x, y), lab in zip(coords, labels):
        if 0 <= x < W and 0 <= y < H:
            roi_mask[y, x] = lab
        else:
            print(f"⚠️ 坐标超出范围: ({x},{y}), 已跳过")

    # 5. 保存
    os.makedirs(os.path.dirname(roi_mask_out), exist_ok=True)
    np.save(roi_mask_out, roi_mask)
    print(f"✅ ROI 掩膜已保存: {roi_mask_out} (shape={roi_mask.shape})")

if __name__ == "__main__":
    # ——— 修改下面三行，指定你的路径 ——— #
    samples_pkl = "../data/samples.pkl"
    ref_img      = "../data/TM_image_AA_preprocessed.tif"
    out_mask     = "../output/ROI/roi_mask.npy"

    generate_roi_mask_from_samples(samples_pkl, ref_img, out_mask)