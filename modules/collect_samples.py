#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
'''
图像分割 - 交互式样本采集模块

功能：在真彩色影像上点击采样并输入类别编号，将采集的坐标和对应特征保存为样本文件，用于监督分类

输入：  
  - 预处理后的真彩色影像（GeoTIFF 格式）  
  - 全图特征数组（NumPy .npy 格式）

输出：  
  - 采样点坐标与标签文件（PKL 格式，(coords, labels)）

作者：Meng Yinan  
日期：2025-05-26
'''

import os
import pickle
import numpy as np
import matplotlib
# **保证使用 GUI 后端**
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.simpledialog import askinteger
from utils.set_chinese_font import set_chinese_font

set_chinese_font()

def normalize_for_display(img):
    """
    对 img 做 2%–98% 分位数拉伸并截断到 [0,1]，只取前三通道。
    """
    # 保证三通道
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    img3 = img[..., :3].astype(np.float32)
    out = np.zeros_like(img3, dtype=np.float32)
    for i in range(3):
        band = img3[..., i]
        lo, hi = np.percentile(band, (2, 98))
        # 如果 lo==hi，就平铺到 0.5
        if hi > lo:
            out[..., i] = np.clip((band - lo) / (hi - lo), 0, 1)
        else:
            out[..., i] = 0.5
    return out

def collect_samples(image_rgb, feature_map, output_path="data/samples.pkl", class_labels=None):
    """
    在 RGB 图像上点击采样，弹窗输入类别编号，采集用于监督分类的样本。
    同时记录每个样本的 (x, y) 坐标，保存为 (coords, labels)。
    """
    if class_labels is None:
        class_labels = {1: "水体", 2: "植被", 3: "建设用地"}

    root = tk.Tk()
    root.withdraw()

    coords, samples, labels = [], [], []

    fig, ax = plt.subplots()
    disp = normalize_for_display(image_rgb)
    ax.imshow(disp)
    ax.set_title(
        "点击采样，关闭窗口结束\n"
        + "类别映射："
        + "  ".join(f"{k}:{v}" for k, v in class_labels.items())
    )

    def onclick(event):
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            prompt = f"为点 ({x},{y}) 输入类别编号 {class_labels}:"
            lab = askinteger(
                "输入类别", prompt,
                parent=root,
                minvalue=min(class_labels), maxvalue=max(class_labels)
            )
            if lab in class_labels:
                coords.append((x, y))
                samples.append(feature_map[y, x, :])
                labels.append(lab)
                ax.plot(x, y, "ro")
                fig.canvas.draw()
            else:
                print(f"❌ 无效编号 {lab}，有效：{list(class_labels.keys())}")

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    root.destroy()

    if not coords:
        print("⚠️ 未采集任何样本，未保存文件。")
        return np.array([]), np.array([]), np.array([])

    coords = np.array(coords, dtype=int)                # N×2
    samples = np.vstack(samples)                        # N×F
    labels = np.array(labels, dtype=int)                # N

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        # 存为 (coords, labels)
        pickle.dump((coords, labels), f)

    print(f"✅ 共采集 {len(coords)} 个样本点，保存至 {output_path}")
    return coords, samples, labels


if __name__ == '__main__':
    import rasterio
    import numpy as np

    # —— 读取真彩色影像 —— #
    ds = rasterio.open('../data/TM_image_AA_preprocessed.png/TM_image_AA_preprocessed.tif')
    print("▶ bands:", ds.count)
    print("▶ dtype:", ds.dtypes)
    print("▶ shape:", ds.height, ds.width)
    image_rgb = np.dstack([ds.read(3), ds.read(2), ds.read(1)])

    # —— 加载特征图 —— #
    feature_map = np.load(
        '../output/feature_outputs/all_hierarchical_features.npy'
    )  # H×W×F

    # —— 调用采样函数 —— #
    coords, samples, labels = collect_samples(
        image_rgb,
        feature_map,
        output_path="../data/samples.pkl"
    )

    # —— 打印 pkl 内容 —— #
    print(f"坐标数组 shape: {coords.shape}")   # (N,2)
    print(f"特征数组 shape: {samples.shape}")  # (N,F)
    print(f"标签数组 shape: {labels.shape}")   # (N,)