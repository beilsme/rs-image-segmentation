#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
'''
图像分割 - 监督分类模块

功能：基于交互采样点和层次化特征使用随机森林训练模型，并对整幅图像进行像素级分类

输入：  
  - 采样坐标与标签文件（PKL格式）  
  - 全图特征数组（NumPy .npy格式）

输出：  
  - 训练好的随机森林模型文件（joblib 或 pickle）  
  - 全图分类结果数组（NumPy .npy格式）  
  - 分类结果可视化图像（PNG格式）

作者：Meng Yinan  
日期：2025-05-26
'''


import os
import pickle
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from utils.set_chinese_font import set_chinese_font
set_chinese_font()

def prepare_training_samples(features, roi_array, target_labels):
    """
    从掩膜图像中提取训练样本（像素级）
    参数:
        features: ndarray (H, W, D)
        roi_array: ndarray (H, W) ，像素级标注掩膜
        target_labels: list，如 [1, 2, 3]
    返回:
        X: 样本特征矩阵 (N, D)
        y: 样本标签向量 (N,)
    """
    try:
        h, w, d = features.shape
        X, y = [], []
        for label in target_labels:
            mask = (roi_array == label)
            rows, cols = np.where(mask)
            for r, c in zip(rows, cols):
                X.append(features[r, c])
                y.append(label)
        return np.array(X), np.array(y)
    except Exception as e:
        print("❌ prepare_training_samples 出错:", e)
        return np.array([]), np.array([])

def train_random_forest(X, y, param_grid=None, save_path="output/rf_model.pkl"):
    """
    使用网格搜索训练随机森林模型
    参数:
        X: 样本特征 (N, D)
        y: 样本标签 (N,)
        param_grid: 超参数字典
    返回:
        训练后的模型（并保存）
    """
    try:
        if param_grid is None:
            param_grid = {
                'n_estimators': [100],
                'max_depth': [10, 20, None],
                'random_state': [42]
            }
        grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, n_jobs=-1)
        grid.fit(X, y)
        best_model = grid.best_estimator_
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(best_model, save_path)
        print(f"✅ 模型训练完成，保存至 {save_path}")
        return best_model
    except Exception as e:
        print("❌ 随机森林训练失败:", e)
        return None

def train_random_forest_from_samples(samples, labels, save_path="output/rf_model.pkl"):
    """
    从交互采样点训练模型（不进行参数搜索）
    """
    try:
        model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
        model.fit(samples, labels)
        joblib.dump(model, save_path)
        print(f"✅ 交互采样模型训练完成，保存至 {save_path}")
        return model
    except Exception as e:
        print("❌ 交互训练失败:", e)
        return None

def predict_image(model, features):
    """
    使用训练好的模型对整幅图像进行像素级分类
    参数:
        model: 已训练的分类器
        features: ndarray (H, W, D)
    返回:
        prediction_map: ndarray (H, W)
    """
    try:
        h, w, d = features.shape
        X_all = features.reshape(-1, d)
        y_all = model.predict(X_all)
        return y_all.reshape(h, w)
    except Exception as e:
        print("❌ 预测失败:", e)
        return np.zeros(features.shape[:2], dtype=int)


if __name__ == '__main__':
    import os
    import numpy as np
    import pickle
    import matplotlib.pyplot as plt

    # ——— 加载采样坐标和标签 ——— #
    samples_pkl = "../data/samples.pkl"
    with open(samples_pkl, "rb") as f:
        coords, labels = pickle.load(f)   # coords: (N,2), labels: (N,)

    # ——— 读取全图特征 ——— #
    fmap_path = "../output/feature_outputs/all_hierarchical_features.npy"
    feature_map = np.load(fmap_path)      # shape (H, W, D)

    # ——— 从 coords 提取真正的特征样本 ——— #
    # coords 存的是 (x,y)，feature_map 用 [y, x, :]
    X = np.array([feature_map[y, x, :] for x, y in coords])
    y = labels

    # ——— 训练模型 ——— #
    model = train_random_forest_from_samples(
        X, y,
        save_path="../output/rf_samples_model.pkl"
    )

    # ——— 全图预测、保存与可视化同原 —— #
    class_map = predict_image(model, feature_map)
    out_dir = "../output"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "class_map.npy"), class_map)
    # ——— 全图可视化并保存 ——— #
    eval_dir = "../output/supervised"
    os.makedirs(eval_dir, exist_ok=True)

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(class_map, cmap="tab10")
    plt.title("全图分类结果")
    plt.axis("off")

    # 保存到指定目录
    save_path = os.path.join(eval_dir, "coarse_supervised_classification_AA.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化结果已保存至: {save_path}")

    plt.show()