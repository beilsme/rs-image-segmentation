#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
图像分割 - 地物分类与后处理模块

功能：基于多种方法（规则、K-Means、随机森林）对影像进行地物分类，支持三分类输出、可视化与 GeoTIFF 保存，并集成高级后处理与指标评估

输入：预处理与特征提取生成的特征文件（.npy/.pkl/.tif），以及可选的带标注 ROI 或监督样本  
输出：分类结果图像（PNG/GeoTIFF）、三分类评估文件、可视化组合指数图、分类报告等

作者：Meng Yinan  
日期：2025-05-26
'''

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report
import cv2
from scipy import ndimage
import os
import rasterio
from rasterio.transform import Affine
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.patches as mpatches # 用于图例


# --- 特征加载 ---
def load_features(file_path):
    """
    加载预处理的特征数据。
    根据文件扩展名自动选择加载方式。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"特征文件未找到: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()
    print(f"正在加载特征文件: {file_path} (格式: {file_ext})")

    features_dict = {}
    if file_ext == '.npy':
        try:
            # allow_pickle=True 以支持可能的字典存储
            features_data = np.load(file_path, allow_pickle=True)
            if features_data.ndim == 0 and features_data.item() and isinstance(features_data.item(), dict):
                # 如果 .npy 文件保存的是一个字典对象
                features_dict = features_data.item()
                print(".npy 文件作为字典加载。")
            elif features_data.ndim == 3: # (bands, height, width)
                # 假设特征按顺序存储，需要根据实际情况调整
                # 这是基于您之前代码的假设
                feature_names_ordered = [f'feature_{i+1}' for i in range(features_data.shape[0])]
                print(f".npy 文件作为3D数组加载，假设顺序为: {feature_names_ordered}")
                temp_f_dict = {}
                for i, name in enumerate(feature_names_ordered):
                    if i < features_data.shape[0]:
                        temp_f_dict[name] = features_data[i]
                    else:
                        break
                # 为了与pkl的结构保持一致，将这些特征放入一个通用键下，或直接用它们填充
                # 这里我们创建一个类似pkl的结构，以便normalize_features_structure可以处理
                features_dict['all_features'] = temp_f_dict
                if features_data.shape[1] > 0 and features_data.shape[2] > 0:
                    features_dict['dimensions'] = (features_data.shape[1], features_data.shape[2]) # H, W
            else:
                raise ValueError(f".npy 文件内容格式未知或不符合预期 (shape: {features_data.shape})。期望字典或 (bands, H, W) 数组。")
        except Exception as e:
            print(f"加载 .npy 文件 '{file_path}' 时出错: {e}")
            raise

    elif file_ext == '.pkl':
        try:
            with open(file_path, 'rb') as f:
                features_dict = pickle.load(f)
        except Exception as e:
            print(f"加载 .pkl 文件 '{file_path}' 时出错: {e}")
            raise

    elif file_ext == '.tif' or file_ext == '.tiff':
        try:
            with rasterio.open(file_path) as src:
                features_array = src.read()
                band_descriptions = src.descriptions
                temp_f_dict = {} # 用于存储特征波段

                if band_descriptions and len(band_descriptions) == features_array.shape[0]:
                    for i, desc in enumerate(band_descriptions):
                        # 使用描述（如果存在且有效）作为键，否则使用通用名称
                        key_name = desc.lower() if desc else f'band_{i+1}'
                        temp_f_dict[key_name] = features_array[i]
                else: # If no descriptions or mismatch, use generic names
                    feature_names_ordered = [f'band_{i+1}' for i in range(features_array.shape[0])]
                    print(f".tif 文件波段描述缺失或不匹配，使用通用名称: {feature_names_ordered}")
                    for i, name in enumerate(feature_names_ordered):
                        if i < features_array.shape[0]:
                            temp_f_dict[name] = features_array[i]
                        else:
                            break

                # Store extracted features and metadata in a structure similar to pkl
                features_dict['all_features'] = temp_f_dict
                features_dict['transform'] = src.transform
                features_dict['crs'] = src.crs
                features_dict['width'] = src.width
                features_dict['height'] = src.height
                features_dict['dimensions'] = (src.height, src.width)

        except Exception as e:
            print(f"加载 .tif 文件 '{file_path}' 时出错: {e}")
            raise
    else:
        raise ValueError(f"Unsupported feature file format: {file_ext} (from file: {file_path})")

    if not features_dict:
        print(f"Warning: Failed to load any features from {file_path}.")
    else:
        print(f"Loaded top-level raw keys: {list(features_dict.keys())}")
    return features_dict

# --- 特征结构规范化 (改进版) ---
def normalize_features_structure(loaded_features):
    """
    规范化加载的特征字典，以确保各个特征数组
    （如 'ndvi', 'ndwi', 或特定的滤波器/纹理波段）
    在顶层可访问，并标准化元数据键。
    改进了处理包含数组的嵌套列表和字典的逻辑
    （例如，容器内的 PCA 列表、GLCM/滤波器字典等）。
    """
    normalized = {}
    print(f"开始特征结构规范化。原始加载的键: {list(loaded_features.keys())}")

    # 使用一个集合来跟踪已经添加的特征键，以避免重复
    added_feature_keys = set()

    # 辅助函数：递归地提取数组
    def extract_arrays(data, prefix=""):
        """
        递归遍历数据结构（字典、列表、数组），提取所有 NumPy 数组并添加到 normalized 字典。
        使用 prefix 构建唯一的键名。
        """
        # 如果是 NumPy 数组且维度大于等于 2
        if isinstance(data, np.ndarray) and data.ndim >= 2:
            # 构建要添加的键名，转换为小写
            key_to_add = prefix.lower()
            # 确保键名非空且尚未添加
            if key_to_add and key_to_add not in added_feature_keys:
                # 将数组添加到 normalized 字典
                normalized[key_to_add] = data
                # 将键名添加到已添加集合中
                added_feature_keys.add(key_to_add)
                # print(f"  已提取数组: '{key_to_add}' (shape: {data.shape})") # 可选：打印提取的数组信息
            # else:
            # print(f"  跳过数组 '{key_to_add}' (键名为空或已添加)。") # 可选：打印跳过信息

        # 如果是字典
        elif isinstance(data, dict):
            # 遍历字典中的每个键值对
            for key, value in data.items():
                # 构建新的前缀：当前前缀_键名
                new_prefix = f"{prefix}_{key}" if prefix else key
                # 递归调用处理值
                extract_arrays(value, new_prefix)

        # 如果是列表
        elif isinstance(data, list):
            # 遍历列表中的每个元素及其索引
            for i, value in enumerate(data):
                # 构建新的前缀：当前前缀_索引
                new_prefix = f"{prefix}_{i}" if prefix else str(i) # 使用索引作为键的一部分
                # 递归调用处理元素
                extract_arrays(value, new_prefix)

        # 忽略其他类型的数据（如数字、字符串、None 等）

    # 1. 首先，提取元数据并直接放入 normalized 字典
    meta_keys_map = {
        'geo_transform': 'transform', # 将 GDAL 的 geo_transform 映射到 rasterio 的 transform
        'crs': 'crs',
        'dimensions': 'dimensions',
        'width': 'width',
        'height': 'height',
        'transform': 'transform' # 如果 transform 直接存在
    }
    for original_key, target_key in meta_keys_map.items():
        # 如果原始键存在且目标键尚未被设置（避免被较低优先级的数据覆盖）
        if original_key in loaded_features and target_key not in normalized:
            if original_key == 'geo_transform' and not isinstance(loaded_features[original_key], Affine):
                # 特殊处理 geo_transform，尝试转换为 Affine 对象
                try:
                    gt = loaded_features[original_key]
                    # 确保 geo_transform 是一个包含 6 个元素的元组或列表
                    if isinstance(gt, (tuple, list)) and len(gt) == 6:
                        normalized[target_key] = Affine.from_gdal(gt[0], gt[1], gt[2], gt[3], gt[4], gt[5])
                        print(f"元数据: '{original_key}' 已作为 Affine 对象转换并复制到 '{target_key}'.")
                    else:
                        print(f"警告: '{original_key}' 格式不正确 ({type(gt)}, len={len(gt) if isinstance(gt,(tuple,list)) else 'N/A'})，无法转换为 Affine 对象。值为: {loaded_features[original_key]}")
                        normalized[target_key] = loaded_features[original_key] # 按原样复制以供调试
                except Exception as e:
                    print(f"警告: 转换 '{original_key}' 到 Affine 对象失败: {e}. 值为: {loaded_features[original_key]}")
                    normalized[target_key] = loaded_features[original_key] # 按原样复制以供调试
            else:
                # 如果不是 geo_transform 或已经是 Affine 对象，直接复制
                normalized[target_key] = loaded_features[original_key]
                print(f"元数据: '{original_key}' 已复制到 '{target_key}'")


    # 2. 使用递归辅助函数提取所有数组
    # 遍历 loaded_features 的顶层键值对
    print("正在从嵌套结构中提取特征数组...")
    for key, value in loaded_features.items():
        lower_key = key.lower()
        # 跳过已经作为元数据处理的键
        if lower_key in meta_keys_map.values():
            # print(f"跳过已知元数据键 '{key}'.")
            continue

        # 对顶层的值调用递归提取函数，使用当前键作为前缀
        # 这会处理顶层的数组、字典、列表，并递归处理其内容
        extract_arrays(value, key)


    # 3. 确定并规范化 'height' 和 'width'
    # 检查 normalized 字典中是否已经有了整数类型的 height 和 width
    if not ('height' in normalized and isinstance(normalized.get('height'), int) and \
            'width' in normalized and isinstance(normalized.get('width'), int)):

        # 如果没有，尝试从 'dimensions' 键中提取
        if 'dimensions' in normalized and isinstance(normalized['dimensions'], tuple):
            dims = normalized['dimensions']
            if len(dims) == 2: # (H, W)
                normalized['height'], normalized['width'] = int(dims[0]), int(dims[1])
            elif len(dims) >= 2: # 假设 (bands, H, W) 或 (H, W, bands)
                # 尝试更鲁棒地确定 H, W (例如，它们通常比波段数大得多)
                if len(dims) == 3 and dims[0] < dims[-2] and dims[0] < dims[-1]: # 可能是 (bands, H, W)
                    normalized['height'], normalized['width'] = int(dims[1]), int(dims[2])
                else: # 否则，假设 (H, W, bands) 或其他，取前两个作为 H,W
                    normalized['height'], normalized['width'] = int(dims[0]), int(dims[1])
            # 检查是否成功提取
            if 'height' in normalized and isinstance(normalized.get('height'), int):
                print(f"从 'dimensions' 键更新了 height: {normalized.get('height')}, width: {normalized.get('width')}.")


    # 如果 height/width 仍然未确定，尝试从已提取的特征数组中推断
    if not ('height' in normalized and isinstance(normalized.get('height'), int) and \
            'width' in normalized and isinstance(normalized.get('width'), int)):
        print("尝试从已提取的特征数组推断 height 和 width...")
        for key, val in normalized.items():
            # 检查任何一个已提取的 NumPy 数组
            if isinstance(val, np.ndarray) and val.ndim >= 2:
                if val.ndim == 2: # (H, W)
                    normalized['height'], normalized['width'] = val.shape
                elif val.ndim == 3 and val.shape[1] > 0 and val.shape[2] > 0: # 假设 (bands, H, W)
                    normalized['height'], normalized['width'] = val.shape[1], val.shape[2]
                elif val.ndim == 3 and val.shape[0] > 0 and val.shape[1] > 0: # 假设 (H, W, bands)
                    normalized['height'], normalized['width'] = val.shape[0], val.shape[1]

                # 确保成功提取
                if 'height' in normalized and isinstance(normalized.get('height'), int):
                    print(f"从特征数组 '{key}' (shape: {val.shape}) 推断了 height: {normalized['height']}, width: {normalized['width']}.")
                    break # 找到后就停止


    # 4. 最终检查和清理
    essential_geo_keys = ['transform', 'crs']
    for eg_key in essential_geo_keys:
        if eg_key not in normalized or normalized[eg_key] is None:
            print(f"警告: 规范化后的特征中可能缺少重要的地理信息: '{eg_key}'")

    if 'height' not in normalized or 'width' not in normalized or \
            not isinstance(normalized.get('height'), int) or not isinstance(normalized.get('width'), int) :
        print("严重警告: 规范化后未能确定有效的图像 'height' 或 'width'。")
    else:
        print(f"规范化后的 height: {normalized['height']}, width: {normalized['width']}")

    # 如果 height/width/transform 都已设置正确，则移除临时的 'dimensions' 和 'geo_transform'
    if 'height' in normalized and 'width' in normalized and 'dimensions' in normalized:
        del normalized['dimensions']
    if 'transform' in normalized and 'geo_transform' in normalized and 'geo_transform' != 'transform':
        if isinstance(normalized.get('transform'), Affine): # 确保 transform 是 Affine 对象才删除 geo_transform
            del normalized['geo_transform']


    print(f"特征结构规范化完成。规范化后的顶层键 ({len(normalized)}): {list(normalized.keys())}")
    # 可选：打印提取的特征键总数和列表，以供验证
    # extracted_feature_keys_count = len(added_feature_keys)
    # print(f"总共提取并添加的特征数组数量: {extracted_feature_keys_count}")
    # if extracted_feature_keys_count > 0:
    #     # 打印前 N 个或全部键，根据需要调整
    #     print(f"已提取的特征数组键列表: {list(added_feature_keys)[:20]}{'...' if extracted_feature_keys_count > 20 else ''}")


    return normalized


# --- 后处理 ---
def advanced_post_processing(binary_mask, min_area=100, smooth_kernel_size=3, fill_holes=True):
    """
    对二值分割掩码进行高级后处理。
    """
    if binary_mask is None or binary_mask.size == 0:
        print("警告: 输入的二值掩码为空，后处理跳过。")
        return binary_mask

    processed_mask = binary_mask.copy().astype(np.uint8)

    # 填充孔洞 (使用闭运算，更鲁棒)
    if fill_holes and smooth_kernel_size > 0 and smooth_kernel_size % 2 == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_kernel_size, smooth_kernel_size))
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    elif fill_holes:
        # Fallback if kernel size is not odd or not specified
        processed_mask = ndimage.binary_fill_holes(processed_mask).astype(np.uint8)


    # 移除小面积区域
    if min_area > 0:
        labeled_mask, num_features = ndimage.label(processed_mask, structure=np.ones((3,3))) # Use 8-connectivity
        if num_features > 0:
            area = np.bincount(labeled_mask.ravel())
            remove_labels = np.where((area < min_area) & (area > 0))[0]
            if remove_labels.size > 0:
                # Create a mask where pixels belonging to small objects are marked True
                remove_pixel_mask = np.isin(labeled_mask, remove_labels)
                processed_mask[remove_pixel_mask] = 0
        else:
            pass # No features found to remove


    # 平滑边界 (形态学开运算) - Applied after hole filling and small object removal
    # This helps to break narrow connections and smooth edges
    if smooth_kernel_size > 0 and smooth_kernel_size % 2 == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_kernel_size, smooth_kernel_size))
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    elif smooth_kernel_size > 0:
        print(f"警告: 平滑核大小 {smooth_kernel_size} 不是奇数，形态学平滑可能效果不佳或出错。")


    return processed_mask

# --- 基于阈值的地物提取 ---
def threshold_segmentation(feature_image, threshold_value, above=True, otsu=False):
    """
    基于阈值的分割方法。
    otsu: 如果为True，则忽略threshold_value，使用Otsu方法自动确定阈值。
    """
    if feature_image is None:
        raise ValueError("输入的特征图像为空。")

    # Handle NaN values - simple fill with 0. Consider more advanced interpolation if needed.
    if np.isnan(feature_image).any():
        # print("Warning: NaN values found in feature image, filling with 0. This might affect thresholding, especially Otsu.")
        feature_image = np.nan_to_num(feature_image, nan=0.0)


    if otsu:
        # Otsu's method requires 8-bit image
        min_val, max_val = np.min(feature_image), np.max(feature_image)
        if max_val == min_val:
            print("警告: 特征图像所有值相同，Otsu无法应用。返回全黑或全白掩码。")
            return np.zeros_like(feature_image, dtype=np.uint8) if above else np.ones_like(feature_image, dtype=np.uint8)

        # Scale to 0-255, clipping values outside the min/max range
        norm_image = np.clip(((feature_image - min_val) / (max_val - min_val + 1e-10) * 255), 0, 255).astype(np.uint8)

        try:
            otsu_thresh_val_norm, mask_otsu = cv2.threshold(norm_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Convert Otsu threshold back to original data range
            otsu_thresh_original = otsu_thresh_val_norm / 255.0 * (max_val - min_val) + min_val
            # print(f"Otsu auto-determined normalized threshold: {otsu_thresh_val_norm}, corresponding original image threshold: {otsu_thresh_original:.4f}")

            mask_otsu = (mask_otsu > 0).astype(np.uint8) # Ensure binary mask (0 or 1)

            if above: # Otsu THRESH_BINARY returns 1 for values > threshold
                return mask_otsu
            else: # Return 1 for values < threshold
                return (1 - mask_otsu).astype(np.uint8)
        except cv2.error as e:
            print(f"OpenCV错误，Otsu阈值处理失败: {e}。可能由于图像内容不适合Otsu（例如，单峰分布）。将回退到基于中位数的阈值。")
            # Fallback logic: If Otsu fails, use a fixed threshold based on the median
            median_thresh = np.median(feature_image)
            print(f"Otsu失败，回退到中位数阈值: {median_thresh:.4f}")
            if above:
                return (feature_image > median_thresh).astype(np.uint8)
            else:
                return (feature_image < median_thresh).astype(np.uint8)

    else:
        if above:
            mask = (feature_image > threshold_value).astype(np.uint8)
        else:
            mask = (feature_image < threshold_value).astype(np.uint8)
        return mask

def extract_vegetation_by_threshold(features_dict, ndvi_threshold=0.2, post_process=True, min_area=100):
    """使用NDVI阈值提取植被"""
    if 'ndvi' not in features_dict or features_dict['ndvi'] is None:
        print("警告: 特征中未找到'ndvi'，无法提取植被。")
        if 'height' in features_dict and 'width' in features_dict:
            return np.zeros((features_dict['height'], features_dict['width']), dtype=np.uint8)
        return np.array([])


    vegetation_mask = threshold_segmentation(features_dict['ndvi'], ndvi_threshold, above=True, otsu=False) # 可以尝试 otsu=True
    if post_process:
        vegetation_mask = advanced_post_processing(vegetation_mask, min_area=min_area, smooth_kernel_size=3)
    return vegetation_mask

def extract_water_by_threshold(features_dict, ndwi_threshold=0.0, mndwi_threshold=0.1, use_mndwi_if_available=True, post_process=True, min_area=50):
    """使用NDWI或MNDWI阈值提取水体"""
    water_mask = None
    used_index = None
    if use_mndwi_if_available and 'mndwi' in features_dict and features_dict['mndwi'] is not None:
        # print("使用MNDWI提取水体")
        water_mask = threshold_segmentation(features_dict['mndwi'], mndwi_threshold, above=True, otsu=False) # 可以尝试 otsu=True
        used_index = 'mndwi'
    elif 'ndwi' in features_dict and features_dict['ndwi'] is not None:
        # print("使用NDWI提取水体")
        water_mask = threshold_segmentation(features_dict['ndwi'], ndwi_threshold, above=True, otsu=False) # 可以尝试 otsu=True
        used_index = 'ndwi'
    else:
        print("警告: 特征中未找到'ndwi'或'mndwi'，无法提取水体。")
        if 'height' in features_dict and 'width' in features_dict:
            return np.zeros((features_dict['height'], features_dict['width']), dtype=np.uint8)
        return np.array([])

    if post_process and water_mask is not None:
        water_mask = advanced_post_processing(water_mask, min_area=min_area, smooth_kernel_size=3)
    return water_mask


def extract_builtup_by_threshold(features_dict, ndbi_threshold=0.0, ndvi_threshold_for_builtup=0.15, post_process=True, min_area=150):
    """使用NDBI阈值并结合NDVI排除植被来提取建设用地"""
    if 'ndbi' not in features_dict or features_dict['ndbi'] is None:
        print("警告: 特征中未找到'ndbi'，无法提取建设用地。")
        if 'height' in features_dict and 'width' in features_dict:
            return np.zeros((features_dict['height'], features_dict['width']), dtype=np.uint8)
        return np.array([])

    builtup_mask_ndbi = threshold_segmentation(features_dict['ndbi'], ndbi_threshold, above=True, otsu=False) # 可以尝试 otsu=True

    if 'ndvi' in features_dict and features_dict['ndvi'] is not None and features_dict['ndvi'].shape == builtup_mask_ndbi.shape:
        # 建设用地通常NDVI较低
        non_veg_mask = threshold_segmentation(features_dict['ndvi'], ndvi_threshold_for_builtup, above=False)
        builtup_mask = np.logical_and(builtup_mask_ndbi, non_veg_mask).astype(np.uint8)
    else:
        builtup_mask = builtup_mask_ndbi
        if 'ndvi' not in features_dict or features_dict['ndvi'] is None:
            print("Warning: 'ndvi' not found or is None. Cannot use NDVI to refine built-up mask.")
        elif features_dict['ndvi'].shape != builtup_mask_ndbi.shape:
            print(f"Warning: NDVI shape {features_dict['ndvi'].shape} does not match NDBI shape {builtup_mask_ndbi.shape}. Cannot use NDVI to refine built-up mask.")


    if post_process:
        builtup_mask = advanced_post_processing(builtup_mask, min_area=min_area, smooth_kernel_size=5) # 建设用地通常较大，用稍大核平滑
    return builtup_mask

def extract_bareland_by_rule(features_dict, vegetation_mask, water_mask, builtup_mask,
                             ndvi_low_threshold=-0.1, ndvi_high_threshold=0.2,
                             ndbi_low_threshold=-0.2, ndbi_high_threshold=0.2,
                             post_process=True, min_area=80):
    """
    通过排除法并结合指数规则提取裸地。
    """
    if 'height' not in features_dict or 'width' not in features_dict:
        print("警告: 提取裸地所需的图像尺寸信息缺失。")
        return np.array([])

    img_h, img_w = features_dict['height'], features_dict['width']
    shape = (img_h, img_w)

    # Initialize exclusion mask
    excluded_mask = np.zeros(shape, dtype=bool)
    if vegetation_mask is not None and vegetation_mask.shape == shape:
        excluded_mask = np.logical_or(excluded_mask, vegetation_mask.astype(bool))
    if water_mask is not None and water_mask.shape == shape:
        excluded_mask = np.logical_or(excluded_mask, water_mask.astype(bool))
    if builtup_mask is not None and builtup_mask.shape == shape:
        excluded_mask = np.logical_or(excluded_mask, builtup_mask.astype(bool))

    potential_bareland_mask = np.logical_not(excluded_mask).astype(np.uint8)

    # Apply index rules to further refine
    if 'ndvi' in features_dict and features_dict['ndvi'] is not None and features_dict['ndvi'].shape == shape:
        ndvi = features_dict['ndvi']
        ndvi_condition = np.logical_and(ndvi > ndvi_low_threshold, ndvi < ndvi_high_threshold)
        potential_bareland_mask = np.logical_and(potential_bareland_mask, ndvi_condition).astype(np.uint8)
    else:
        print("警告: 裸地提取规则缺少NDVI或形状不匹配。")


    if 'ndbi' in features_dict and features_dict['ndbi'] is not None and features_dict['ndbi'].shape == shape:
        ndbi = features_dict['ndbi']
        ndbi_condition = np.logical_and(ndbi > ndbi_low_threshold, ndbi < ndbi_high_threshold)
        potential_bareland_mask = np.logical_and(potential_bareland_mask, ndbi_condition).astype(np.uint8)
    else:
        print("警告: 裸地提取规则缺少NDBI或形状不匹配。")


    if post_process:
        potential_bareland_mask = advanced_post_processing(potential_bareland_mask, min_area=min_area, smooth_kernel_size=3)

    return potential_bareland_mask

# --- 无监督分类 ---
def unsupervised_kmeans_classification(features_dict, n_clusters=5, feature_keys_to_use=None):
    """使用K-Means进行无监督分类"""
    if not features_dict or 'height' not in features_dict or 'width' not in features_dict:
        raise ValueError("特征字典为空或缺少图像尺寸信息 (height/width)。")

    img_h, img_w = features_dict['height'], features_dict['width']
    img_shape_tuple = (img_h, img_w)

    if feature_keys_to_use is None:
        # Automatically select available numerical features (2D NumPy arrays)
        non_feature_metadata_keys = ['transform', 'crs', 'width', 'height', 'dimensions', 'geo_transform']
        available_feature_keys = [
            k for k, v in features_dict.items()
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape == img_shape_tuple and k not in non_feature_metadata_keys
        ]
        # If too few features are automatically selected, can specify a default set
        if not available_feature_keys:
            default_candidates = ['ndvi', 'ndwi', 'ndbi', 'texture_mean', 'evi', 'savi',
                                  'hierarchical_level_1', 'hierarchical_level_2', 'hierarchical_all'] # Added hierarchical
            available_feature_keys = [k for k in default_candidates if k in features_dict and \
                                      isinstance(features_dict[k], np.ndarray) and \
                                      (features_dict[k].ndim == 2 and features_dict[k].shape == img_shape_tuple or features_dict[k].ndim == 3 and features_dict[k].shape[:2] == img_shape_tuple)]

        feature_keys_to_use = available_feature_keys

    if not feature_keys_to_use:
        raise ValueError("没有可用于K-Means的特征。请检查特征字典内容或手动指定 `feature_keys_to_use`。")

    print(f"用于K-Means的特征: {feature_keys_to_use}")

    # Prepare data
    feature_list_for_stacking = []
    for key in feature_keys_to_use:
        if key in features_dict and features_dict[key] is not None:
            feature_img = features_dict[key]
            # Handle potential 3D arrays if hierarchical features are directly selected
            if isinstance(feature_img, np.ndarray) and feature_img.ndim == 3 and feature_img.shape[:2] == img_shape_tuple: # (H, W, bands)
                # Flatten each band and append
                for i in range(feature_img.shape[2]):
                    flat_band = feature_img[:,:,i].flatten()
                    if np.isnan(flat_band).any():
                        # print(f"Warning: Feature '{key}' (band {i}) contains NaN values, filling with 0.")
                        flat_band = np.nan_to_num(flat_band, nan=0.0)
                    feature_list_for_stacking.append(flat_band)
            elif isinstance(feature_img, np.ndarray) and feature_img.ndim == 2 and feature_img.shape == img_shape_tuple: # (H, W)
                flat_band = feature_img.flatten()
                if np.isnan(flat_band).any():
                    # print(f"Warning: Feature '{key}' contains NaN values, filling with 0.")
                    flat_band = np.nan_to_num(flat_band, nan=0.0)
                feature_list_for_stacking.append(flat_band)
            else:
                print(f"Warning: Feature '{key}' has unexpected shape {feature_img.shape if isinstance(feature_img, np.ndarray) else 'N/A'} for K-Means, skipping.")
        else:
            # This error should not happen as feature_keys_to_use should only contain valid keys
            print(f"Severe Warning: Feature '{key}' does not exist or is None in the dictionary, despite being in feature_keys_to_use.")


    if not feature_list_for_stacking:
        raise ValueError("未能准备任何特征数据进行K-Means分类。")

    data_for_kmeans = np.vstack(feature_list_for_stacking).T # (n_pixels, n_features)

    # Standardize
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_for_kmeans)

    # K-Means clustering
    print(f"正在进行K-Means聚类，目标簇数: {n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto', verbose=0)
    labels = kmeans.fit_predict(data_scaled)

    classification_result = labels.reshape(img_shape_tuple)
    print("K-Means聚类完成。")
    return classification_result


# --- 监督分类 (示例：随机森林) ---
def prepare_training_samples(feature_array, labeled_roi_path):
    """
    从带有标签的ROI（栅格文件）中提取训练样本，使用预先堆叠的特征数组。

    Args:
        feature_array (np.ndarray): 整个影像的堆叠特征数组 (height, width, n_features)。
        labeled_roi_path (str): 带有标签的ROI栅格文件路径。

    Returns:
        tuple: (X_samples, y_labels)，其中 X_samples 是用于训练的特征矩阵，
               y_labels 是对应的标签向量。
    """
    if not os.path.exists(labeled_roi_path):
        raise FileNotFoundError(f"标签ROI文件未找到: {labeled_roi_path}")

    if feature_array is None or feature_array.ndim != 3:
        raise ValueError("输入的feature_array必须是3D NumPy数组 (height, width, n_features)。")

    img_h, img_w, n_features = feature_array.shape
    expected_shape_2d = (img_h, img_w)

    with rasterio.open(labeled_roi_path) as src:
        labels_img = src.read(1)
        if labels_img.shape != expected_shape_2d:
            raise ValueError(f"标签ROI文件 '{labeled_roi_path}' (形状 {labels_img.shape}) 尺寸与特征数据 (期望的2D形状 {expected_shape_2d}) 不匹配。")

    X_samples = []
    y_labels = []

    # Flatten feature array for easier indexing
    feature_array_flat = feature_array.reshape(-1, n_features)
    labels_img_flat = labels_img.flatten()

    # Extract samples based on non-zero labels
    valid_indices = np.where((labels_img_flat != 0) & (~np.isnan(labels_img_flat)))[0]

    if valid_indices.size == 0:
        raise ValueError("未能从ROI中提取任何有效的训练样本。请检查标签ROI文件和类别定义。")

    X_samples = feature_array_flat[valid_indices, :]
    y_labels = labels_img_flat[valid_indices]

    # Handle potential NaN values in extracted samples (should ideally be handled in feature extraction, but double check)
    nan_in_samples = np.isnan(X_samples).any()
    if nan_in_samples:
        print("警告: 提取的训练样本中存在NaN值，将用0填充。")
        X_samples = np.nan_to_num(X_samples, nan=0.0)

    return X_samples, y_labels

def train_random_forest_classifier(X_train, y_train, feature_names_for_training, n_estimators=100, test_size=0.3, random_state=42):
    """训练随机森林分类器并进行评估"""
    # Ensure there are enough samples per class for stratification if needed
    unique_classes, counts = np.unique(y_train, return_counts=True)
    if len(unique_classes) > 1 and np.min(counts) < 2:
        print(f"警告: 部分类别样本数少于2个 ({dict(zip(unique_classes, counts))})。训练/验证分割时这些类别可能无法进行分层抽样。")
        # Adjust stratify parameter if necessary, or handle classes with < 2 samples
        stratify_param = y_train if len(unique_classes) > 1 and np.min(counts) >= 2 else None
    else:
        stratify_param = y_train if len(unique_classes) > 1 else None


    X_t, X_val, y_t, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state, stratify=stratify_param)

    print(f"训练样本数: {X_t.shape[0]}, 验证样本数: {X_val.shape[0]}")
    if X_t.shape[0] == 0:
        raise ValueError("没有足够的训练样本进行分割。")

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    print("正在训练随机森林分类器...")
    clf.fit(X_t, y_t)

    # Evaluation
    if X_val.shape[0] > 0:
        y_pred_val = clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred_val)
        kappa = cohen_kappa_score(y_val, y_pred_val)
        print(f"随机森林分类器验证集性能:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  Kappa系数: {kappa:.4f}")
        print("\n  分类报告:")
        # Ensure all unique labels from both y_val and y_pred_val are included in the report
        report_labels = np.unique(np.concatenate((y_val, y_pred_val)))
        print(classification_report(y_val, y_pred_val, labels=report_labels, zero_division=0))

        # Feature Importance
        if len(feature_names_for_training) == clf.feature_importances_.size:
            importances = clf.feature_importances_
            sorted_indices = np.argsort(importances)[::-1]
            print("\n  特征重要性:")
            for i in sorted_indices:
                # Ensure index is within bounds of feature_names_for_training
                if i < len(feature_names_for_training):
                    print(f"    特征 '{feature_names_for_training[i]}': {importances[i]:.4f}")
                else:
                    print(f"    特征 (Index {i}): {importances[i]:.4f} (特征名称列表索引超出范围)")
        else:
            print("\n警告: 特征重要性无法显示。特征名称数量与分类器的特征重要性数组大小不匹配。")


    else:
        print("警告: 验证样本数为0，跳过验证评估。")

    return clf

def supervised_classification_predict(feature_array, classifier):
    """
    使用训练好的分类器对整个影像进行分类，使用预先堆叠的特征数组。

    Args:
        feature_array (np.ndarray): 整个影像的堆叠特征数组 (height, width, n_features)。
        classifier: 训练好的scikit-learn分类器对象 (例如, RandomForestClassifier)。

    Returns:
        np.ndarray: 预测的分类图 (height, width)。
    """
    if feature_array is None or feature_array.ndim != 3:
        raise ValueError("输入的feature_array必须是3D NumPy数组 (height, width, n_features)。")

    img_h, img_w, n_features = feature_array.shape

    # Reshape feature array for prediction
    data_to_predict = feature_array.reshape(-1, n_features)

    # Handle potential NaN values in prediction data
    if np.isnan(data_to_predict).any():
        print("警告: 预测数据中存在NaN值，将用0填充。")
        data_to_predict = np.nan_to_num(data_to_predict, nan=0.0)


    print("正在对整个影像进行分类...")
    classification_result_flat = classifier.predict(data_to_predict)
    classification_result = classification_result_flat.reshape((img_h, img_w))
    print("监督分类完成。")
    return classification_result


# --- 可视化与保存 ---
def create_classification_map(classification_result, class_names_map, class_colors_map, save_path="classification_map.png", title="地物分类图"):
    """
    创建并保存彩色分类图。
    """
    if classification_result is None or classification_result.size == 0:
        print("分类结果为空，无法创建地图。")
        return None

    height, width = classification_result.shape
    rgb_map = np.zeros((height, width, 3), dtype=np.uint8)

    unique_classes = np.unique(classification_result)
    # print(f"分类结果中包含的类别ID: {unique_classes}")

    legend_patches = []

    # Create legend entries only for classes actually present in the result and defined in color map
    active_class_ids = sorted([int(cid) for cid in unique_classes if cid in class_colors_map]) # Ensure integer keys

    for class_id in active_class_ids:
        rgb_map[classification_result == class_id] = class_colors_map[class_id]
        class_name = class_names_map.get(class_id, f"类别 {class_id}")
        # Create patch with color normalized to [0, 1]
        legend_patches.append(mpatches.Patch(color=[c/255.0 for c in class_colors_map[class_id]], label=class_name))

    # Warn about classes in result but not in color map
    for class_id in unique_classes:
        if class_id not in class_colors_map:
            print(f"警告: 类别ID {class_id} 在分类结果中存在，但在class_colors_map中未定义颜色，将显示为黑色。")


    plt.figure(figsize=(12, 10))
    plt.imshow(rgb_map)
    plt.title(title, fontsize=16)
    plt.axis('off')

    if legend_patches:
        plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
    else:
        print("警告: 未能为分类图生成图例。")

    # Adjust layout to ensure legend is fully visible
    plt.subplots_adjust(right=0.75 if legend_patches else 0.95)


    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"分类图已保存至: {save_path}")
        except Exception as e:
            print(f"保存分类图 '{save_path}' 失败: {e}")

    plt.show()
    return rgb_map

def save_classification_as_geotiff(classification_result, features_meta, output_tif_path):
    """
    将分类结果保存为GeoTIFF文件，使用规范化后的特征元数据。
    features_meta: 包含 'transform', 'crs', 'width', 'height' 的字典。
    """
    if classification_result is None or classification_result.size == 0:
        print("分类结果为空，无法保存为GeoTIFF。")
        return

    if not all(k in features_meta and features_meta[k] is not None for k in ['transform', 'crs', 'width', 'height']):
        print(f"警告: 用于保存GeoTIFF的元数据不完整。需要 'transform', 'crs', 'width', 'height'。")
        print(f"可用的元数据键: {list(features_meta.keys())}")
        return

    try:
        # Ensure classification result is of an appropriate integer type for GeoTIFF
        # Convert to uint8 if possible, otherwise use uint16 or int32
        if np.max(classification_result) <= 255 and np.min(classification_result) >= 0:
            output_dtype = rasterio.uint8
        elif np.max(classification_result) <= 65535 and np.min(classification_result) >= 0:
            output_dtype = rasterio.uint16
        else:
            output_dtype = rasterio.int32 # Or handle larger ranges if necessary

        # Handle potential float labels from intermediate processing
        if np.issubdtype(classification_result.dtype, np.floating):
            print("警告: 分类结果包含浮点数标签，将转换为整数。")
            classification_result = np.round(classification_result).astype(output_dtype)
        else:
            classification_result = classification_result.astype(output_dtype)


        profile = {
            'driver': 'GTiff',
            'dtype': output_dtype,
            'nodata': 0, # Assuming 0 is unclassified/background, can set as nodata
            'width': features_meta['width'],
            'height': features_meta['height'],
            'count': 1,
            'crs': features_meta['crs'],
            'transform': features_meta['transform'],
            'compress': 'lzw',
            'tiled': True, # Add tiling for better performance with large images
            'blockxsize': 256,
            'blockysize': 256
        }

        # Ensure output shape matches metadata
        if classification_result.shape[0] != profile['height'] or classification_result.shape[1] != profile['width']:
            print(f"Warning: Classification result shape {classification_result.shape} does not match dimensions in metadata ({profile['height']}, {profile['width']}). Skipping GeoTIFF save.")
            return


        with rasterio.open(output_tif_path, 'w', **profile) as dst:
            dst.write(classification_result, 1) # Use already converted dtype
        print(f"分类结果已保存为GeoTIFF: {output_tif_path}")

    except Exception as e:
        print(f"保存GeoTIFF文件 '{output_tif_path}' 时出错: {e}")


# --- 可视化特定指数组合 (增加图例版) ---
def visualize_combined_indices(features_dict, output_dir="visualization_outputs", save_path="combined_indices_map.png"):
    """
    从特征字典中提取水体、植被、土壤相关的指数特征，并合成一张彩色图进行可视化，并添加图例。
    此版本兼容经过新 normalize_features_structure 函数规范化后的带有前缀的键名。
    通常使用 NDWI/MNDWI (水体), NDVI/EVI/MSAVI (植被), BSI (裸土) 或其他相关指数。
    这里我们尝试组合 水体(蓝色通道), 植被(绿色通道), 土壤/建筑(红色通道) 相关指数。
    """
    os.makedirs(output_dir, exist_ok=True)
    full_save_path = os.path.join(output_dir, save_path)

    print("\n--- 可视化水体、植被、土壤特征组合图 ---")

    # 定义用于RGB通道的特征键名列表 (优先顺序从左到右)
    # 注意：这里使用经过 normalize_features_structure 处理后可能产生的键名
    # 优先从 all_extracted_features_dict 中查找，然后是顶层键
    blue_candidates = [
        'all_extracted_features_dict_mndwi', 'all_extracted_features_dict_ndwi',
        'mndwi', 'ndwi'
    ]
    green_candidates = [
        'all_extracted_features_dict_evi', 'all_extracted_features_dict_msavi', 'all_extracted_features_dict_ndvi',
        'evi', 'msavi', 'ndvi'
    ]
    red_candidates = [
        'all_extracted_features_dict_bsi', 'all_extracted_features_dict_ndbi',
        'bsi', 'ndbi'
    ]

    # 查找实际存在于 features_dict 中的特征
    blue_channel_feature_name = None
    for key in blue_candidates:
        if key in features_dict and isinstance(features_dict.get(key), np.ndarray) and features_dict.get(key) is not None and features_dict.get(key).ndim >= 2:
            blue_channel_feature_name = key
            break

    green_channel_feature_name = None
    for key in green_candidates:
        if key in features_dict and isinstance(features_dict.get(key), np.ndarray) and features_dict.get(key) is not None and features_dict.get(key).ndim >= 2:
            green_channel_feature_name = key
            break

    red_channel_feature_name = None
    for key in red_candidates:
        if key in features_dict and isinstance(features_dict.get(key), np.ndarray) and features_dict.get(key) is not None and features_dict.get(key).ndim >= 2:
            red_channel_feature_name = key
            break


    # 获取特征图像
    blue_feature = features_dict.get(blue_channel_feature_name) if blue_channel_feature_name else None
    green_feature = features_dict.get(green_channel_feature_name) if green_channel_feature_name else None
    red_feature = features_dict.get(red_channel_feature_name) if red_channel_feature_name else None


    # 检查是否有足够的特征进行组合 (至少需要3个通道对应的特征)
    feature_list_for_rgb = [f for f in [red_feature, green_feature, blue_feature] if f is not None and f.ndim >= 2]
    feature_names_for_rgb = [n for n, f in zip([red_channel_feature_name, green_channel_feature_name, blue_channel_feature_name], [red_feature, green_feature, blue_feature]) if f is not None and f.ndim >= 2]


    if len(feature_list_for_rgb) < 3:
        print(f"警告: 无法找到足够的（至少3个）特征进行RGB组合可视化。尝试查找的特征: 蓝色通道: {blue_candidates}, 绿色通道: {green_candidates}, 红色通道: {red_candidates}")
        print(f"实际找到的有效特征: {feature_names_for_rgb}")

        if not feature_list_for_rgb:
            print("错误: 没有找到任何有效的特征数组进行可视化。")
            return None

        # 如果找到少于3个特征，尝试使用可用的特征创建灰度图或伪彩色图
        print("将使用找到的可用的特征创建灰度图或伪彩色图。")
        # Simple average of available features
        # Ensure all features have the same shape before stacking
        img_shape = None
        if 'height' in features_dict and 'width' in features_dict:
            img_shape = (features_dict['height'], features_dict['width'])

        valid_features_same_shape = []
        valid_feature_names_same_shape = []
        if img_shape:
            for i, f_array in enumerate(feature_list_for_rgb):
                # Check shape and remove extra dimensions if they are 1 (e.g., (H, W, 1))
                if f_array.ndim >= 2 and f_array.shape[:2] == img_shape:
                    if f_array.ndim == 3 and f_array.shape[2] == 1:
                        valid_features_same_shape.append(np.squeeze(f_array))
                    elif f_array.ndim == 2:
                        valid_features_same_shape.append(f_array)
                    else:
                        print(f"警告: 特征 '{feature_names_for_rgb[i]}' 形状 {f_array.shape} 不适合灰度图组合 (需要 2D 或 (H, W, 1))。")
                        continue
                    valid_feature_names_same_shape.append(feature_names_for_rgb[i])
                else:
                    print(f"警告: 特征 '{feature_names_for_rgb[i]}' 形状 {f_array.shape} 与图像尺寸 {img_shape} 不符，跳过。")

        if not valid_features_same_shape:
            print("错误: 没有找到与图像尺寸匹配的有效特征数组进行可视化。")
            return None

        # Stack valid features and calculate mean
        stacked_valid_features = np.stack(valid_features_same_shape, axis=-1)

        # Handle potential NaN values
        if np.isnan(stacked_valid_features).any():
            # print("警告: 用于灰度图组合的特征包含NaN值，将用中位数填充。") # 或用 0 填充
            # Option 1: Fill with mean/median
            # nan_mask = np.isnan(stacked_valid_features)
            # col_mean = np.nanmean(stacked_valid_features, axis=(0,1))
            # stacked_valid_features = np.where(nan_mask, col_mean, stacked_valid_features)
            # Option 2: Fill with 0 (simpler for visualization)
            print("警告: 用于灰度图组合的特征包含NaN值，将用0填充。")
            stacked_valid_features = np.nan_to_num(stacked_valid_features, nan=0.0)


        combined_grayscale = np.mean(stacked_valid_features, axis=-1)


        # Normalize to [0, 1]
        min_val = np.min(combined_grayscale)
        max_val = np.max(combined_grayscale)
        if max_val == min_val:
            print("警告: 组合特征灰度图所有值相同。")
            combined_grayscale_norm = np.zeros_like(combined_grayscale)
        else:
            combined_grayscale_norm = (combined_grayscale - min_val) / (max_val - min_val + 1e-10)

        plt.figure(figsize=(10, 8))
        plt.imshow(combined_grayscale_norm, cmap='gray') # Or another suitable colormap like 'viridis'
        plt.colorbar(label='组合特征强度', fraction=0.046, pad=0.04)
        plt.title('可用的水体/植被/土壤相关特征组合 (灰度)', fontsize=14)
        plt.xlabel(f"组合了特征: {', '.join(valid_feature_names_same_shape)}")
        plt.axis('off')

        # Save the grayscale map
        try:
            save_path_grayscale = full_save_path.replace('.png', '_grayscale.png')
            plt.savefig(save_path_grayscale, dpi=300, bbox_inches='tight')
            print(f"组合特征灰度图已保存至: {save_path_grayscale}")
        except Exception as e:
            print(f"保存组合特征灰度图失败: {e}")

        plt.show()
        return None # Return None as RGB image was not created


    # 确保三个选中的特征图像的尺寸一致
    # Note: We already checked f.ndim >= 2, now check full shape after squeezing if needed
    # Squeeze features that might have shape (H, W, 1)
    red_feature = np.squeeze(red_feature) if red_feature.ndim == 3 and red_feature.shape[2] == 1 else red_feature
    green_feature = np.squeeze(green_feature) if green_feature.ndim == 3 and green_feature.shape[2] == 1 else green_feature
    blue_feature = np.squeeze(blue_feature) if blue_feature.ndim == 3 and blue_feature.shape[2] == 1 else blue_feature


    if not (red_feature.shape == green_feature.shape == blue_feature.shape):
        print(f"错误: 用于组合的特征图像尺寸不一致 (挤压单波段后)。红通道('{red_channel_feature_name}'): {red_feature.shape}, 绿通道('{green_channel_feature_name}'): {green_feature.shape}, 蓝通道('{blue_channel_feature_name}'): {blue_feature.shape}")
        # Attempt to resize if dimensions match (H, W)
        img_shape = (features_dict['height'], features_dict['width'])
        resize_needed = False
        features_to_resize = []
        feature_names_check = [red_channel_feature_name, green_channel_feature_name, blue_channel_feature_name] # Use actual names for messages
        for i, f_array in enumerate([red_feature, green_feature, blue_feature]):
            if f_array.shape != img_shape:
                print(f"警告: 特征 '{feature_names_check[i]}' 形状 {f_array.shape} 与图像尺寸 {img_shape} 不符，将尝试调整。")
                try:
                    # Resize to (W, H) order for cv2
                    resized_f = cv2.resize(f_array.astype(np.float32), (img_shape[1], img_shape[0]), interpolation=cv2.INTER_CUBIC) # Use cubic for continuous data
                    features_to_resize.append(resized_f)
                    resize_needed = True
                except Exception as e_resize:
                    print(f"调整特征 '{feature_names_check[i]}' 大小失败: {e_resize}. 无法创建RGB组合图。")
                    return None
            else:
                features_to_resize.append(f_array)

        if resize_needed:
            red_feature, green_feature, blue_feature = features_to_resize
            print("特征已调整到一致尺寸。")
        else: # Shapes didn't match and no resize was needed (shouldn't happen here)
            return None


    # Handle potential NaN values before normalization
    if np.isnan(red_feature).any() or np.isnan(green_feature).any() or np.isnan(blue_feature).any():
        print("警告: 用于RGB组合的特征包含NaN值，将用0填充。")
        red_feature = np.nan_to_num(red_feature, nan=0.0)
        green_feature = np.nan_to_num(green_feature, nan=0.0)
        blue_feature = np.nan_to_num(blue_feature, nan=0.0)


    # 对每个特征进行归一化到 [0, 1] 范围，以便可视化
    def normalize_for_vis(band):
        # Simple Min-Max normalization to [0, 1]
        min_val = np.min(band)
        max_val = np.max(band)
        if max_val == min_val: return np.zeros_like(band) # Avoid division by zero
        return (band - min_val) / (max_val - min_val + 1e-10)

    blue_norm = normalize_for_vis(blue_feature)
    green_norm = normalize_for_vis(green_feature)
    red_norm = normalize_for_vis(red_feature)


    # 将归一化后的特征堆叠为 RGB 图像 (注意顺序: 红、绿、蓝通道)
    # 我们想让水体偏蓝，植被偏绿，土壤/建筑偏红。
    # 所以将 Soil/Built-up 放在 R 通道，Vegetation 放在 G 通道，Water 放在 B 通道。
    combined_rgb = np.dstack((red_norm, green_norm, blue_norm))

    # 裁剪值到 [0, 1] 范围
    combined_rgb = np.clip(combined_rgb, 0, 1)


    # 创建并显示图像
    plt.figure(figsize=(12, 10))
    plt.imshow(combined_rgb)
    plt.title('水体/植被/土壤相关特征组合图', fontsize=16)
    # 显示通道使用的特征名称 (作为xlabel或标题的一部分)
    plt.xlabel(f"红通道: {red_channel_feature_name.upper()}, 绿通道: {green_channel_feature_name.upper()}, 蓝通道: {blue_channel_feature_name.upper()}")
    plt.axis('off')

    # --- 添加图例 ---
    legend_patches = []
    # 创建代表预期颜色的 Patch
    # 预期的水体颜色 (蓝色通道)
    legend_patches.append(mpatches.Patch(color=[0/255.0, 0/255.0, 255/255.0], label=f'预期的水体颜色 ({blue_channel_feature_name.upper()})'))
    # 预期的植被颜色 (绿色通道)
    legend_patches.append(mpatches.Patch(color=[0/255.0, 255/255.0, 0/255.0], label=f'预期的植被颜色 ({green_channel_feature_name.upper()})'))
    # 预期的土壤/建筑颜色 (红色通道)
    legend_patches.append(mpatches.Patch(color=[255/255.0, 0/255.0, 0/255.0], label=f'预期的土壤/建筑颜色 ({red_channel_feature_name.upper()})'))
    # 您可以根据需要添加更多颜色的描述，例如黄色的裸地 (红+绿):
    # legend_patches.append(mpatches.Patch(color=[255/255.0, 255/255.0, 0/255.0], label='预期的裸地颜色 (高红+高绿)'))


    if legend_patches:
        # 将图例放置在图像外部，避免遮挡图像
        plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)

    # 调整布局以确保图例完全可见
    plt.subplots_adjust(right=0.75 if legend_patches else 0.95)


    # 保存图像
    try:
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        print(f"组合特征图已保存至: {full_save_path}")
    except Exception as e:
        print(f"保存组合特征图失败: {e}")

    plt.show()

    return combined_rgb
