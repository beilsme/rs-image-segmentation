#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
图像分割 - 遥感影像特征提取与增强模块

功能：对预处理后的遥感影像计算各类光谱指数（NDVI、EVI、MSAVI、NDWI、MNDWI、NDBI、BSI）、
执行主成分分析（PCA）、提取多种纹理（GLCM、LBP、Gabor）、形态学与滤波器响应，并支持
多尺度与空间上下文特征构建。

输入：预处理后的波段数据数组列表  
输出：各类特征数组（NumPy）与可视化图像  

Author: Meng Yinan  
Date: 2025-05-26
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def robust_normalize(band, lower_percentile=2, upper_percentile=98):
    """
    使用稳健归一化方法处理波段数据，避免异常值影响
    
    参数:
        band: 输入波段数据
        lower_percentile: 下限百分位数，默认为2%
        upper_percentile: 上限百分位数，默认为98%
    
    返回:
        normalized_band: 归一化后的波段数据，范围[0,1]
    """
    # 计算百分位数
    min_val = np.percentile(band, lower_percentile)
    max_val = np.percentile(band, upper_percentile)

    # 截断异常值
    band_clipped = np.clip(band, min_val, max_val)

    # 归一化到[0,1]
    epsilon = 1e-10  # 防止除零
    normalized_band = (band_clipped - min_val) / (max_val - min_val + epsilon)

    return normalized_band

def calculate_ndvi(nir_band, red_band):
    """
    计算归一化植被指数(NDVI)
    
    参数:
        nir_band: 近红外波段(TM波段4)
        red_band: 红光波段(TM波段3)
    
    返回:
        ndvi: 归一化植被指数
    """
    # 防止除零错误
    denominator = nir_band + red_band
    mask = denominator > 0.001  # 使用小阈值而非严格的零

    ndvi = np.zeros_like(nir_band, dtype=np.float32)
    ndvi[mask] = (nir_band[mask] - red_band[mask]) / denominator[mask]

    # NDVI范围为[-1, 1]，将异常值截断
    ndvi = np.clip(ndvi, -1.0, 1.0)

    return ndvi

def calculate_evi(nir_band, red_band, blue_band, L=1, C1=6, C2=7.5, G=2.5):
    """
    计算增强型植被指数(EVI)，对大气和土壤背景的影响更加稳健
    
    参数:
        nir_band: 近红外波段
        red_band: 红光波段
        blue_band: 蓝光波段
        L, C1, C2, G: EVI参数
    
    返回:
        evi: 增强型植被指数
    """
    denominator = nir_band + C1 * red_band - C2 * blue_band + L
    mask = denominator > 0.001  # 使用小阈值

    evi = np.zeros_like(nir_band, dtype=np.float32)
    evi[mask] = G * (nir_band[mask] - red_band[mask]) / denominator[mask]

    # 通常EVI范围在[-1, 1]之间，但可能略有超出
    evi = np.clip(evi, -1.0, 1.0)

    return evi

def calculate_msavi(nir_band, red_band):
    """
    计算修正土壤调整植被指数(MSAVI)，减少土壤背景影响
    
    参数:
        nir_band: 近红外波段
        red_band: 红光波段
    
    返回:
        msavi: 修正土壤调整植被指数
    """
    # MSAVI2公式
    msavi = (2 * nir_band + 1 - np.sqrt((2 * nir_band + 1)**2 - 8 * (nir_band - red_band))) / 2

    # 范围通常在[-1, 1]
    msavi = np.clip(msavi, -1.0, 1.0)

    return msavi

def calculate_ndwi(green_band, nir_band):
    """
    计算归一化水体指数(NDWI)
    
    参数:
        green_band: 绿光波段(TM波段2)
        nir_band: 近红外波段(TM波段4)
    
    返回:
        ndwi: 归一化水体指数
    """
    # 防止除零错误
    denominator = green_band + nir_band
    mask = denominator > 0.001  # 使用小阈值

    ndwi = np.zeros_like(green_band, dtype=np.float32)
    ndwi[mask] = (green_band[mask] - nir_band[mask]) / denominator[mask]

    # NDWI范围为[-1, 1]
    ndwi = np.clip(ndwi, -1.0, 1.0)

    return ndwi

def calculate_mndwi(green_band, swir_band):
    """
    计算改进的归一化水体指数(MNDWI)，对水体识别更敏感
    
    参数:
        green_band: 绿光波段(TM波段2)
        swir_band: 短波红外波段(TM波段5)
    
    返回:
        mndwi: 改进的归一化水体指数
    """
    denominator = green_band + swir_band
    mask = denominator > 0.001

    mndwi = np.zeros_like(green_band, dtype=np.float32)
    mndwi[mask] = (green_band[mask] - swir_band[mask]) / denominator[mask]

    mndwi = np.clip(mndwi, -1.0, 1.0)

    return mndwi

def calculate_ndbi(swir_band, nir_band):
    """
    计算归一化建筑指数(NDBI)
    
    参数:
        swir_band: 短波红外波段(TM波段5)
        nir_band: 近红外波段(TM波段4)
    
    返回:
        ndbi: 归一化建筑指数
    """
    denominator = swir_band + nir_band
    mask = denominator > 0.001

    ndbi = np.zeros_like(swir_band, dtype=np.float32)
    ndbi[mask] = (swir_band[mask] - nir_band[mask]) / denominator[mask]

    ndbi = np.clip(ndbi, -1.0, 1.0)

    return ndbi

def calculate_bsi(blue_band, red_band, nir_band, swir_band):
    """
    计算裸土指数(BSI)，更好地识别裸露土壤
    
    参数:
        blue_band: 蓝光波段
        red_band: 红光波段
        nir_band: 近红外波段
        swir_band: 短波红外波段
    
    返回:
        bsi: 裸土指数
    """
    numerator = (swir_band + red_band) - (nir_band + blue_band)
    denominator = (swir_band + red_band) + (nir_band + blue_band)
    mask = denominator > 0.001

    bsi = np.zeros_like(blue_band, dtype=np.float32)
    bsi[mask] = numerator[mask] / denominator[mask]

    bsi = np.clip(bsi, -1.0, 1.0)

    return bsi

def perform_pca(bands_data, n_components=None, use_robust_scaling=True):
    """
    执行主成分分析，带有稳健数据预处理
    
    参数:
        bands_data: 包含所有波段数据的列表
        n_components: 保留的主成分数量，默认为None(保留所有)
        use_robust_scaling: 是否使用稳健缩放预处理
    
    返回:
        pca_result: 主成分分析结果
        explained_variance_ratio: 各主成分解释的方差比例
        pca: PCA对象，用于后续转换新数据
    """
    # 获取数据维度
    height, width = bands_data[0].shape
    n_bands = len(bands_data)

    # 重塑数据为(pixel_count, band_count)的形状
    X = np.zeros((height * width, n_bands), dtype=np.float32)
    for i in range(n_bands):
        X[:, i] = bands_data[i].flatten()

    # 数据预处理: 稳健缩放或标准缩放
    if use_robust_scaling:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        # 简单的最大最小归一化
        X_scaled = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0) + 1e-10)

    # 执行PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # 将结果重塑回原始图像形状
    pca_result = []
    for i in range(X_pca.shape[1]):
        component = X_pca[:, i].reshape(height, width)
        pca_result.append(component)

    return pca_result, pca.explained_variance_ratio_, pca

def calculate_glcm_features(band, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=32, window_size=21, step_size=21):
    """
    计算灰度共生矩阵(GLCM)纹理特征
    
    参数:
        band: 输入的波段数据
        distances: 像素距离列表
        angles: 角度列表
        levels: 灰度级别数量
        window_size: 纹理窗口大小
        step_size: 窗口移动步长
    
    返回:
        glcm_features: 包含多种GLCM纹理特征的字典
    """
    # 确保输入数据在[0,1]范围内
    band = robust_normalize(band)

    # 转换为需要的灰度级别
    band_scaled = (band * (levels - 1)).astype(np.uint8)

    # 初始化输出特征
    height, width = band.shape
    out_height = (height - window_size) // step_size + 1
    out_width = (width - window_size) // step_size + 1

    # 初始化特征图
    contrast_img = np.zeros((out_height, out_width), dtype=np.float32)
    dissimilarity_img = np.zeros_like(contrast_img)
    homogeneity_img = np.zeros_like(contrast_img)
    energy_img = np.zeros_like(contrast_img)
    correlation_img = np.zeros_like(contrast_img)

    # 滑动窗口计算纹理特征
    for i in range(0, height - window_size + 1, step_size):
        for j in range(0, width - window_size + 1, step_size):
            window = band_scaled[i:i+window_size, j:j+window_size]

            # 计算GLCM
            glcm = graycomatrix(window, distances=distances, angles=angles,
                                levels=levels, symmetric=True, normed=True)

            # 计算纹理特征
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()

            # 存储到结果图中
            out_i = i // step_size
            out_j = j // step_size
            contrast_img[out_i, out_j] = contrast
            dissimilarity_img[out_i, out_j] = dissimilarity
            homogeneity_img[out_i, out_j] = homogeneity
            energy_img[out_i, out_j] = energy
            correlation_img[out_i, out_j] = correlation

    # 将小尺寸特征图重采样到原始大小
    resize_img = lambda img: cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

    glcm_features = {
        'contrast': resize_img(contrast_img),
        'dissimilarity': resize_img(dissimilarity_img),
        'homogeneity': resize_img(homogeneity_img),
        'energy': resize_img(energy_img),
        'correlation': resize_img(correlation_img)
    }

    return glcm_features

def calculate_lbp_features(band, radius=3, n_points=24):
    """
    计算局部二值模式(LBP)特征
    
    参数:
        band: 输入的波段数据
        radius: LBP半径
        n_points: 采样点数量
    
    返回:
        lbp: LBP特征图
    """
    # 确保输入数据在[0,1]范围内
    band = robust_normalize(band)

    # 转换为uint8类型(LBP需要)
    band_uint8 = (band * 255).astype(np.uint8)

    # 计算LBP
    lbp = local_binary_pattern(band_uint8, n_points, radius, method='uniform')

    # 归一化LBP特征
    lbp = lbp / lbp.max()

    return lbp

def calculate_gabor_features(band, num_scales=4, num_orientations=6):
    """
    计算Gabor滤波器纹理特征
    
    参数:
        band: 输入的波段数据
        num_scales: 尺度数量
        num_orientations: 方向数量
    
    返回:
        gabor_features: Gabor特征列表
    """
    # 确保输入数据在[0,1]范围内
    band = robust_normalize(band)

    # 转换为uint8类型
    band_uint8 = (band * 255).astype(np.uint8)

    # Gabor参数
    scales = np.logspace(-1, 0.5, num=num_scales)
    orientations = np.arange(0, np.pi, np.pi / num_orientations)

    # 初始化特征列表
    gabor_features = []

    # 生成Gabor特征
    for scale in scales:
        for theta in orientations:
            # 创建Gabor滤波器
            kernel_size = int(5 * scale)
            if kernel_size % 2 == 0:
                kernel_size += 1
            if kernel_size < 5:
                kernel_size = 5

            gabor_kernel = cv2.getGaborKernel(
                (kernel_size, kernel_size),
                sigma=scale,
                theta=theta,
                lambd=10*scale,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F
            )

            # 应用Gabor滤波器
            filtered_img = cv2.filter2D(band_uint8, cv2.CV_32F, gabor_kernel)

            # 归一化特征
            filtered_img = (filtered_img - filtered_img.min()) / (filtered_img.max() - filtered_img.min() + 1e-10)

            gabor_features.append(filtered_img)

    return gabor_features

def calculate_morphological_features(band):
    """
    计算形态学特征
    
    参数:
        band: 输入的波段数据
    
    返回:
        morphological_features: 形态学特征字典
    """
    # 确保输入数据在[0,1]范围内
    band = robust_normalize(band)

    # 转换为uint8类型
    band_uint8 = (band * 255).astype(np.uint8)

    # 定义形态学操作的核
    kernel_sizes = [3, 5, 7]
    features = {}

    for size in kernel_sizes:
        kernel = np.ones((size, size), np.uint8)

        # 腐蚀
        erosion = cv2.erode(band_uint8, kernel, iterations=1)
        # 膨胀
        dilation = cv2.dilate(band_uint8, kernel, iterations=1)
        # 开运算(腐蚀后膨胀)
        opening = cv2.morphologyEx(band_uint8, cv2.MORPH_OPEN, kernel)
        # 闭运算(膨胀后腐蚀)
        closing = cv2.morphologyEx(band_uint8, cv2.MORPH_CLOSE, kernel)
        # 形态学梯度(膨胀减腐蚀)
        gradient = cv2.morphologyEx(band_uint8, cv2.MORPH_GRADIENT, kernel)

        # 归一化特征
        features[f'erosion_{size}'] = erosion / 255.0
        features[f'dilation_{size}'] = dilation / 255.0
        features[f'opening_{size}'] = opening / 255.0
        features[f'closing_{size}'] = closing / 255.0
        features[f'gradient_{size}'] = gradient / 255.0

    return features

def calculate_filter_responses(band):
    """
    计算多种滤波器响应
    
    参数:
        band: 输入的波段数据
    
    返回:
        filter_features: 滤波器特征字典
    """
    # 确保输入数据在[0,1]范围内
    band = robust_normalize(band)

    # 转换为uint8类型
    band_uint8 = (band * 255).astype(np.uint8)

    features = {}

    # 高斯滤波
    gaussian_5 = cv2.GaussianBlur(band_uint8, (5, 5), 0) / 255.0
    gaussian_15 = cv2.GaussianBlur(band_uint8, (15, 15), 0) / 255.0
    features['gaussian_5'] = gaussian_5
    features['gaussian_15'] = gaussian_15

    # DoG (Difference of Gaussians)
    dog = gaussian_5 - gaussian_15
    features['dog'] = (dog - dog.min()) / (dog.max() - dog.min() + 1e-10)

    # 拉普拉斯滤波
    laplacian = cv2.Laplacian(band_uint8, cv2.CV_32F) / 255.0
    features['laplacian'] = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min() + 1e-10)

    # Sobel滤波(梯度)
    sobel_x = cv2.Sobel(band_uint8, cv2.CV_32F, 1, 0) / 255.0
    sobel_y = cv2.Sobel(band_uint8, cv2.CV_32F, 0, 1) / 255.0
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    features['sobel_mag'] = sobel_mag / (sobel_mag.max() + 1e-10)

    return features

def feature_selection_by_variance(features_dict, threshold=0.01):
    """
    基于方差的特征选择
    
    参数:
        features_dict: 特征字典
        threshold: 方差阈值，低于此值的特征将被移除
    
    返回:
        selected_features: 选择后的特征字典
    """
    selected_features = {}

    for name, feature in features_dict.items():
        # 计算特征方差
        if isinstance(feature, np.ndarray) and feature.ndim == 2:
            variance = np.var(feature)
            if variance >= threshold:
                selected_features[name] = feature
        # 对于特征列表或字典，递归检查
        elif isinstance(feature, list) and all(isinstance(f, np.ndarray) for f in feature):
            selected_list = []
            for i, f in enumerate(feature):
                var = np.var(f)
                if var >= threshold:
                    selected_list.append(f)
            if selected_list:
                selected_features[name] = selected_list
        elif isinstance(feature, dict):
            sub_selected = {k: v for k, v in feature.items() if isinstance(v, np.ndarray) and np.var(v) >= threshold}
            if sub_selected:
                selected_features[name] = sub_selected

    return selected_features

def calculate_multi_scale_features(band, scales=[1, 3, 5, 7]):
    """
    计算多尺度特征
    
    参数:
        band: 输入的波段数据
        scales: 尺度列表，表示不同的窗口大小
    
    返回:
        multi_scale_features: 多尺度特征字典
    """
    # 确保输入数据在[0,1]范围内
    band = robust_normalize(band)

    features = {}

    for scale in scales:
        # 均值滤波(平滑度)
        mean = cv2.blur(band, (scale, scale))
        features[f'mean_scale_{scale}'] = mean

        # 方差(局部对比度)
        mean_sq = cv2.blur(band * band, (scale, scale))
        variance = mean_sq - mean * mean
        variance[variance < 0] = 0  # 修正浮点数计算可能导致的负值
        features[f'variance_scale_{scale}'] = variance

        # 标准差
        std_dev = np.sqrt(variance)
        features[f'std_dev_scale_{scale}'] = std_dev

        # 局部熵 (使用固定窗口计算)
        if scale <= 5:  # 限制窗口大小以提高效率
            from skimage.filters.rank import entropy
            from skimage.morphology import disk

            # 转换为uint8
            band_uint8 = (band * 255).astype(np.uint8)
            entropy_img = entropy(band_uint8, disk(scale))
            # 归一化熵图
            entropy_img = entropy_img / np.max(entropy_img)
            features[f'entropy_scale_{scale}'] = entropy_img

    return features

def visualize_selected_features(features_dict, max_features=12, save_path="selected_features_visualization.png"):
    """
    可视化选定的特征
    
    参数:
        features_dict: 特征字典
        max_features: 最多显示的特征数量
        save_path: 保存路径
    """
    # 扁平化特征字典
    flat_features = {}
    for key, value in features_dict.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            flat_features[key] = value
        elif isinstance(value, list) and all(isinstance(f, np.ndarray) for f in value):
            for i, f in enumerate(value):
                flat_features[f"{key}_{i}"] = f
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray) and sub_value.ndim == 2:
                    flat_features[f"{key}_{sub_key}"] = sub_value

    # 如果特征太多，选择前max_features个
    feature_names = list(flat_features.keys())
    if len(feature_names) > max_features:
        feature_names = feature_names[:max_features]

    # 计算子图布局
    n_features = len(feature_names)
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=(4 * n_cols, 3 * n_rows))

    for i, name in enumerate(feature_names):
        plt.subplot(n_rows, n_cols, i + 1)

        # 获取特征数据
        feature = flat_features[name]

        # 归一化显示
        feature_norm = (feature - np.min(feature)) / (np.max(feature) - np.min(feature) + 1e-10)

        # 选择合适的颜色映射
        if 'ndvi' in name.lower():
            cmap = 'RdYlGn'
        elif 'ndwi' in name.lower() or 'water' in name.lower():
            cmap = 'Blues'
        elif 'ndbi' in name.lower() or 'build' in name.lower():
            cmap = 'hot'
        elif 'pca' in name.lower():
            cmap = 'viridis'
        elif 'texture' in name.lower() or 'glcm' in name.lower() or 'lbp' in name.lower():
            cmap = 'gray'
        else:
            cmap = 'viridis'

        plt.imshow(feature_norm, cmap=cmap)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(name)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def feature_fusion_for_segmentation(features_dict, selected_features=None, fusion_method='weighted_sum'):
    """
    特征融合，为分割准备
    
    参数:
        features_dict: 特征字典
        selected_features: 需要融合的特征名称列表，None表示全部
        fusion_method: 融合方法，可选'weighted_sum'或'concatenate'
    
    返回:
        fused_features: 融合后的特征
    """
    # 如果未指定需要融合的特征，则使用所有特征
    if selected_features is None:
        # 只选择2D数组特征
        selected_features = [name for name, feature in features_dict.items()
                             if isinstance(feature, np.ndarray) and feature.ndim == 2]

    # 获取选定的特征
    features_to_fuse = []
    for name in selected_features:
        if name in features_dict and isinstance(features_dict[name], np.ndarray) and features_dict[name].ndim == 2:
            # 归一化特征
            feature = features_dict[name]
            feature_norm = (feature - np.min(feature)) / (np.max(feature) - np.min(feature) + 1e-10)
            features_to_fuse.append(feature_norm)

    # 确保有特征要融合
    if not features_to_fuse:
        raise ValueError("没有有效的特征可以融合")

    # 融合特征
    if fusion_method == 'weighted_sum':
        # 默认所有特征权重相等
        weights = np.ones(len(features_to_fuse)) / len(features_to_fuse)
        fused_feature = np.zeros_like(features_to_fuse[0])
        for i, feature in enumerate(features_to_fuse):
            fused_feature += weights[i] * feature

    elif fusion_method == 'concatenate':
        # 对于分割任务，通常需要将特征在通道维度上堆叠
        # 返回形状为(height, width, n_features)的数组
        fused_feature = np.stack(features_to_fuse, axis=-1)

    else:
        raise ValueError(f"不支持的融合方法: {fusion_method}")

    return fused_feature

def prepare_features_for_segmentation(features_dict, important_features=None):
    """
    为地物分割准备特征集
    
    参数:
        features_dict: 特征字典
        important_features: 重要特征列表，None表示使用所有适合的特征
    
    返回:
        segmentation_features: 用于分割的特征数组，形状为(height, width, n_features)
    """
    if important_features is None:
        # 默认使用指数类特征和纹理特征
        important_features = []
        for name in features_dict.keys():
            # 选择指数类特征
            if any(index in name.lower() for index in ['ndvi', 'evi', 'msavi', 'ndwi', 'mndwi', 'ndbi', 'bsi']):
                important_features.append(name)
            # 选择主成分
            elif 'pca' in name.lower() and isinstance(features_dict[name], list):
                # 只选择前三个主成分
                for i in range(min(3, len(features_dict[name]))):
                    important_features.append(f"{name}_{i}")

    # 构建特征数组
    feature_arrays = []

    for name in important_features:
        if name in features_dict and isinstance(features_dict[name], np.ndarray) and features_dict[name].ndim == 2:
            # 直接添加2D特征
            feature = features_dict[name]
            feature_norm = robust_normalize(feature)
            feature_arrays.append(feature_norm)
        elif '_' in name:
            # 处理如"pca_result_0"的特征名
            base_name, idx = name.rsplit('_', 1)
            if base_name in features_dict and isinstance(features_dict[base_name], list):
                try:
                    idx = int(idx)
                    if 0 <= idx < len(features_dict[base_name]):
                        feature = features_dict[base_name][idx]
                        feature_norm = robust_normalize(feature)
                        feature_arrays.append(feature_norm)
                except ValueError:
                    continue

    # 堆叠特征
    if feature_arrays:
        segmentation_features = np.stack(feature_arrays, axis=-1)
        return segmentation_features
    else:
        raise ValueError("没有找到适合分割的特征")

def hierarchical_feature_fusion(features_dict):
    """
    分层次融合特征，先关注主要类别区分，再关注细分类
    """
    # 第一级特征 - 关注主要类别区分（水体、植被、建筑、裸地）
    level1_features = []

    # 水体特征（同时适用于河流和湖泊）
    water_features = [
        features_dict['ndwi'],  # 水体敏感
        features_dict['mndwi']  # 改进的水体指数
    ]

    # 植被特征（同时适用于山地和城市植被）
    vegetation_features = [
        features_dict['ndvi'],  # 植被敏感
        features_dict['evi']    # 增强型植被指数
    ]

    # 整合所有一级特征
    level1_features.extend(water_features)
    level1_features.extend(vegetation_features)
    # 添加建筑和裸地特征
    level1_features.append(features_dict['ndbi'])
    level1_features.append(features_dict['bsi'])

    return np.stack(level1_features, axis=-1)

def add_spatial_context(features_array, window_size=7):
    """
    添加空间上下文信息以改进分类
    """
    height, width, n_features = features_array.shape
    context_features = np.zeros((height, width, n_features))

    for i in range(n_features):
        feature = features_array[:,:,i]
        # 添加平均空间上下文
        context = cv2.boxFilter(feature, -1, (window_size, window_size),
                                normalize=True, borderType=cv2.BORDER_REFLECT)
        context_features[:,:,i] = context

    # 连接原始特征和上下文特征
    enhanced_features = np.concatenate([features_array, context_features], axis=-1)
    return enhanced_features

def semantic_merge_water_classes(segmentation_result):
    """
    将分割结果中的河流和湖泊类别合并为水体
    """
    # 假设河流为类别1，湖泊为类别2
    river_mask = (segmentation_result == 1)
    lake_mask = (segmentation_result == 2)

    # 创建合并后的结果
    merged_result = segmentation_result.copy()
    # 将所有水体类型都设为同一类别（例如类别1）
    merged_result[river_mask | lake_mask] = 1

    return merged_result

def evaluate_feature_importance_for_classes(features, training_samples):
    """
    评估各特征对区分主要类别的重要性
    """
    from sklearn.ensemble import RandomForestClassifier

    # 训练简单的随机森林模型
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(features, training_samples)

    # 获取特征重要性
    importance = clf.feature_importances_

    return importance

def prepare_level_1_features(features_dict):
    """
    准备用于主要类别区分的第一级特征
    (水体、植被、建筑/城市、裸土)
    """
    # 选择对主要类别区分效果好的特征
    level_1_features = []

    # 水体特征 (适用于所有水体类型)
    level_1_features.append(features_dict['ndwi'])  # 水体敏感
    level_1_features.append(features_dict['mndwi']) # 改进的水体指数

    # 植被特征 (适用于所有植被类型)
    level_1_features.append(features_dict['ndvi'])  # 植被敏感
    level_1_features.append(features_dict['evi'])   # 增强型植被指数，对土壤背景不敏感

    # 建筑和城市特征
    level_1_features.append(features_dict['ndbi'])  # 建筑指数

    # 裸土特征
    level_1_features.append(features_dict['bsi'])   # 裸土指数

    # 添加第一个主成分 (总体地物区分)
    if 'pca_result' in features_dict and len(features_dict['pca_result']) > 0:
        level_1_features.append(features_dict['pca_result'][0])

    # 堆叠特征
    return np.stack(level_1_features, axis=-1)

def prepare_level_2_features(features_dict):
    """
    准备用于细分类的第二级特征
    (区分河流/湖泊, 山地植被/城市植被等)
    """
    level_2_features = []

    # 纹理特征 (对区分河流/湖泊和不同植被类型有帮助)
    if 'glcm_features' in features_dict:
        # 仅选择对细分类有用的纹理特征
        level_2_features.append(features_dict['glcm_features']['contrast'])
        level_2_features.append(features_dict['glcm_features']['homogeneity'])

    # 形态学特征 (捕捉地物形状差异)
    if 'morphological_features' in features_dict:
        if 'gradient_5' in features_dict['morphological_features']:
            level_2_features.append(features_dict['morphological_features']['gradient_5'])

    # 添加多尺度特征 (捕捉尺度差异, 如河流vs湖泊)
    if 'multi_scale_features' in features_dict:
        if 'std_dev_scale_5' in features_dict['multi_scale_features']:
            level_2_features.append(features_dict['multi_scale_features']['std_dev_scale_5'])

    # 滤波器响应 (边缘和纹理信息)
    if 'filter_features' in features_dict and 'sobel_mag' in features_dict['filter_features']:
        level_2_features.append(features_dict['filter_features']['sobel_mag'])

    # 堆叠特征
    return np.stack(level_2_features, axis=-1) if level_2_features else np.zeros((1,1,1))

def visualize_hierarchical_features(hierarchical_features, features_dict):
    """
    可视化层次化特征
    """
    # 第一级特征可视化 (主要类别区分)
    plt.figure(figsize=(15, 10))
    plt.suptitle('第一级特征 - 主要类别区分', fontsize=16)

    level_1 = hierarchical_features['level_1']
    n_features = min(6, level_1.shape[2])
    for i in range(n_features):
        plt.subplot(2, 3, i+1)
        plt.imshow(level_1[:,:,i], cmap='viridis')
        plt.title(f'特征 L1-{i+1}')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('level_1_features.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 第二级特征可视化 (细分类)
    if hierarchical_features['level_2'].shape[2] > 1:
        plt.figure(figsize=(15, 10))
        plt.suptitle('第二级特征 - 细分类', fontsize=16)

        level_2 = hierarchical_features['level_2']
        n_features = min(6, level_2.shape[2])
        for i in range(n_features):
            plt.subplot(2, 3, i+1)
            plt.imshow(level_2[:,:,i], cmap='plasma')
            plt.title(f'特征 L2-{i+1}')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('level_2_features.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 特征组合效果展示
    plt.figure(figsize=(15, 5))
    plt.suptitle('层次特征组合效果', fontsize=16)

    # 展示水体特征 (NDWI & MNDWI的组合)
    plt.subplot(131)
    water_feature = (features_dict['ndwi'] + features_dict['mndwi']) / 2
    plt.imshow(water_feature, cmap='Blues')
    plt.title('水体特征组合')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    # 展示植被特征 (NDVI & EVI的组合)
    plt.subplot(132)
    vegetation_feature = (features_dict['ndvi'] + features_dict['evi']) / 2
    plt.imshow(vegetation_feature, cmap='Greens')
    plt.title('植被特征组合')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    # 展示建筑和裸土特征
    plt.subplot(133)
    urban_feature = (features_dict['ndbi'] + features_dict['bsi']) / 2
    plt.imshow(urban_feature, cmap='OrRd')
    plt.title('建筑与裸土特征')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('combined_features.png', dpi=300, bbox_inches='tight')
    plt.close()
