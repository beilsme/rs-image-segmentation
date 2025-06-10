#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
'''
图像分割 - 遥感图像分类模块

功能：基于提取的特征对影像进行地物分类，支持规则分类、K-Means聚类和随机森林监督分类，本次实验使用K-Means聚类算法进行分类。

输入：特征数据文件(PKL格式)
输出：分类结果图像和栅格数据

作者：Meng Yinan
日期：2025-05-26
'''


from modules.utils.set_chinese_font import set_chinese_font

set_chinese_font()

import joblib
from affine import Affine
import rasterio
from rasterio.crs import CRS

from modules.features.extract import *
from modules.utils.set_chinese_font import set_chinese_font

set_chinese_font()



def create_three_class_map(classification_result, method='rule_based'):
    """
    将多类分类结果转换为三分类结果
    
    参数:
    classification_result: 原始分类结果数组
    method: 分类方法类型，用于确定类别映射规则
    
    返回:
    三分类结果数组 (1-水体, 2-植被, 3-建设用地, 0-未分类)
    """
    three_class_map = np.zeros_like(classification_result, dtype=np.uint8)

    if method == 'rule_based':
        # 基于规则分类的映射
        # 原始编码：1-植被, 2-水体, 3-建设用地, 4-裸地
        three_class_map[classification_result == 2] = 1  # 水体
        three_class_map[classification_result == 1] = 2  # 植被
        three_class_map[classification_result == 3] = 3  # 建设用地
        # 裸地(4)归入未分类(0)

    elif method == 'kmeans':
        # K-Means结果需要人工解释各簇的含义
        # 这里提供一个示例映射，实际使用时需要根据具体结果调整
        print("警告：K-Means结果需要手动解释各簇含义，请根据实际情况调整映射规则")

        # 示例映射（需要根据实际分析结果调整）
        # 假设簇1-3对应水体，簇4-6对应植被，簇7-10对应建设用地
        water_clusters = [1, 2]  # 根据实际情况调整
        vegetation_clusters = [3, 4, 5]  # 根据实际情况调整
        builtup_clusters = [6, 7]  # 根据实际情况调整

        for cluster in water_clusters:
            three_class_map[classification_result == cluster] = 1
        for cluster in vegetation_clusters:
            three_class_map[classification_result == cluster] = 2
        for cluster in builtup_clusters:
            three_class_map[classification_result == cluster] = 3

    elif method == 'random_forest':
        # 随机森林结果直接使用训练时的类别标签
        three_class_map[classification_result == 2] = 1  # 水体
        three_class_map[classification_result == 1] = 2  # 植被
        three_class_map[classification_result == 3] = 3  # 建设用地

    return three_class_map

def save_three_class_evaluation_tif(classification_map, features_meta, output_path, method='rule_based'):
    """
    保存用于精度评估的三分类GeoTIFF文件
    
    参数:
    classification_map: 分类结果数组
    features_meta: 包含地理元数据的字典
    output_path: 输出文件路径
    method: 分类方法
    """

    # 转换为三分类结果
    three_class_map = create_three_class_map(classification_map, method)

    # 获取图像尺寸
    height, width = three_class_map.shape

    # 处理空间参考信息
    transform = None
    crs = None

    # 获取仿射变换矩阵
    if 'transform' in features_meta and features_meta['transform'] is not None:
        transform = features_meta['transform']
        if not isinstance(transform, Affine):
            # 如果是GDAL格式的地理变换，转换为Affine格式
            if hasattr(transform, '__iter__') and len(transform) == 6:
                transform = Affine.from_gdal(*transform)
    elif 'geo_transform' in features_meta and features_meta['geo_transform'] is not None:
        geo_transform = features_meta['geo_transform']
        if hasattr(geo_transform, '__iter__') and len(geo_transform) == 6:
            transform = Affine.from_gdal(*geo_transform)

    # 获取坐标参考系统
    if 'crs' in features_meta and features_meta['crs'] is not None:
        crs_info = features_meta['crs']
        try:
            if hasattr(crs_info, 'to_wkt'):
                crs = crs_info
            elif isinstance(crs_info, str):
                crs = CRS.from_string(crs_info)
            else:
                crs = CRS.from_string(str(crs_info))
        except Exception as e:
            print(f"警告：无法解析坐标参考系统: {e}")
            crs = None

    # 构建输出参数
    output_profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': rasterio.uint8,
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512
    }

    if transform is not None:
        output_profile['transform'] = transform
    if crs is not None:
        output_profile['crs'] = crs

    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 写入GeoTIFF文件
    with rasterio.open(output_path, 'w', **output_profile) as dst:
        dst.write(three_class_map.astype(rasterio.uint8), 1)

        # 设置颜色映射表
        colormap = {
            0: (0, 0, 0, 255),        # 未分类 - 黑色
            1: (0, 0, 255, 255),      # 水体 - 蓝色
            2: (0, 128, 0, 255),      # 植被 - 绿色
            3: (255, 0, 0, 255)       # 建设用地 - 红色
        }
        dst.write_colormap(1, colormap)

        # 设置波段描述
        dst.set_band_description(1, "Land Cover Classification (1=Water, 2=Vegetation, 3=Built-up)")

    # 计算并打印统计信息
    unique_values, counts = np.unique(three_class_map, return_counts=True)
    total_pixels = three_class_map.size

    class_names = {0: '未分类', 1: '水体', 2: '植被', 3: '建设用地'}

    print(f"✅ 三分类精度评估文件已保存: {output_path}")
    print("分类统计结果:")
    for value, count in zip(unique_values, counts):
        percentage = (count / total_pixels) * 100
        class_name = class_names.get(value, f'未知类别{value}')
        print(f"  {class_name} (类别{value}): {count:,} 像素 ({percentage:.2f}%)")

def run_three_class_evaluation_output(feature_file_path, method='rule_based', output_dir="evaluation_outputs"):
    """
    运行三分类精度评估输出流程
    
    参数:
    feature_file_path: 特征文件路径
    method: 分类方法 ('rule_based', 'kmeans', 'random_forest')
    output_dir: 输出目录
    """
    from modules.features.extract import load_features, normalize_features_structure

    print(f"开始生成三分类精度评估文件，方法: {method}")
    print("="*50)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 加载特征数据
        raw_features = load_features(feature_file_path)
        if not raw_features:
            print("错误：加载特征数据失败")
            return

        features = normalize_features_structure(raw_features)

        # 验证数据完整性
        if 'height' not in features or 'width' not in features:
            print("错误：缺少图像尺寸信息")
            return

        img_shape = (features['height'], features['width'])
        print(f"图像尺寸: {img_shape}")

        # 根据方法执行分类（这里简化处理，实际应调用相应的分类函数）
        # 您需要根据实际情况调用相应的分类函数获取classification_result

        # 示例：假设已有分类结果
        # 在实际使用中，您需要替换为实际的分类结果
        print("注意：这是示例代码，您需要将其集成到实际的分类流程中")

        # 示例分类结果（实际使用时替换为真实结果）
        example_classification = np.random.randint(0, 5, img_shape, dtype=np.uint8)

        # 生成三分类评估文件
        output_filename = f"{method}_three_class_evaluation.tif"
        output_path = os.path.join(output_dir, output_filename)

        save_three_class_evaluation_tif(
            classification_map=example_classification,
            features_meta=features,
            output_path=output_path,
            method=method
        )

        print("="*50)
        print("三分类精度评估文件生成完成")

    except Exception as e:
        print(f"生成三分类评估文件时发生错误: {e}")
        import traceback
        traceback.print_exc()

# 集成到原有代码的修改建议
def integrate_three_class_output_to_main_workflow():
    """
    将三分类输出集成到主工作流程的修改建议
    """
    modification_guide = """
    要将三分类输出集成到您的主工作流程中，请在 run_classification_stage 函数末尾添加以下代码：
    
    # 在现有代码的最后，保存GeoTIFF部分之后添加：
    
    # 生成用于精度评估的三分类TIF文件
    if final_classification_map is not None and final_classification_map.size > 0:
        evaluation_output_path = os.path.join(output_dir, f"{method}_three_class_evaluation.tif")
        save_three_class_evaluation_tif(
            classification_map=final_classification_map,
            features_meta=features,
            output_path=evaluation_output_path,
            method=method
        )
        print(f"三分类精度评估文件已生成: {evaluation_output_path}")
    
    这样，每次运行分类后都会自动生成用于精度评估的三分类TIF文件。
    """
    print(modification_guide)



# --- 主流程与示例 ---
def run_classification_stage(feature_file_path, method='rule_based', output_dir="segmentation_outputs", use_hierarchical_all=True):
    """
    地物分割主工作流程。
    method: 'rule_based', 'kmeans', 'random_forest'
    use_hierarchical_all (bool): For 'random_forest' method, attempt to use the 'hierarchical_all' feature array if available.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"开始地物分割流程，方法: {method}")
    print("="*30)

    # 1. 加载并规范化特征
    try:
        raw_features = load_features(feature_file_path)
        if not raw_features:
            print("加载原始特征失败。")
            return

        # Note: Normalization is done here to make features accessible for rule-based and KMeans,
        # but for Random Forest with hierarchical features, we might use the raw hierarchical array directly.
        features = normalize_features_structure(raw_features)

        # Check if normalized features contain valid image data and metadata
        has_array_data = any(isinstance(v, np.ndarray) and v.ndim >= 2 for k,v in features.items()
                             if k not in ['transform', 'crs', 'width', 'height', 'dimensions', 'geo_transform',
                                          'hierarchical_level_1', 'hierarchical_level_2', 'hierarchical_all']) # Exclude stacked hierarchical arrays here
        # Also check if hierarchical features were successfully loaded and normalized
        has_hierarchical_all = 'hierarchical_all' in features and isinstance(features['hierarchical_all'], np.ndarray) and features['hierarchical_all'].ndim == 3


        has_dims = 'height' in features and 'width' in features and \
                   isinstance(features['height'], int) and isinstance(features['width'], int)

        if not (has_array_data or has_hierarchical_all) or not has_dims:
            print("错误：规范化后的特征不包含有效的图像数组数据（单个特征或hierarchical_all）或尺寸信息。")
            print(f"规范化后的键: {list(features.keys())}")
            if has_dims: print(f"  尺寸: H={features.get('height')}, W={features.get('width')}")
            else: print("  尺寸信息缺失。")
            return

    except Exception as e:
        print(f"加载或规范化特征失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # Get image shape from normalized features
    img_shape = (features['height'], features['width'])
    print(f"工作流程将使用图像形状: {img_shape}")


    # 定义类别和颜色 (根据实际需求调整)
    # 类别ID从1开始，0通常保留给未分类或背景
    class_names = {
        0: '未分类', 1: '植被', 2: '水体', 3: '建设用地', 4: '裸地',
        5: 'KMeans 簇 5', 6: 'KMeans 簇 6', 7: 'KMeans 簇 7', # 为KMeans结果预留
        8: 'KMeans 簇 8', 9: 'KMeans 簇 9', 10: 'KMeans 簇 10'
    }
    class_colors = {
        0: [0, 0, 0],       1: [0, 128, 0],   2: [0, 0, 255],
        3: [255, 0, 0],     4: [255, 255, 0], 5: [128, 0, 128],
        6: [0, 255, 255],   7: [255, 165, 0], 8: [128, 128, 128],
        9: [0, 128, 128], 10: [128, 128, 0]
    }

    final_classification_map = np.zeros(img_shape, dtype=np.uint8)
    title = "地物分类结果" # 默认标题

    if method == 'rule_based':
        print("\n--- 基于规则的分类 ---")
        # 注意：阈值需要根据具体数据调整
        veg_mask = extract_vegetation_by_threshold(features, ndvi_threshold=0.25, min_area=int(img_shape[0]*img_shape[1]*0.0005))
        water_mask = extract_water_by_threshold(features, ndwi_threshold=0.05, min_area=int(img_shape[0]*img_shape[1]*0.0002))
        builtup_mask = extract_builtup_by_threshold(features, ndbi_threshold=0.0, ndvi_threshold_for_builtup=0.2, min_area=int(img_shape[0]*img_shape[1]*0.001))

        # Ensure masks are not None and have correct shape (functions should return correct shape or empty array)
        masks_to_check = {'veg': veg_mask, 'water': water_mask, 'builtup': builtup_mask}
        valid_masks = {}
        for m_name, m_array in masks_to_check.items():
            if m_array is not None and m_array.shape == img_shape:
                valid_masks[m_name] = m_array
            elif m_array is not None and m_array.size > 0:
                print(f"警告: {m_name} 掩码形状 {m_array.shape} 与期望 {img_shape} 不符，将尝试调整。")
                try:
                    valid_masks[m_name] = cv2.resize(m_array.astype(np.uint8), (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
                except Exception as e_resize:
                    print(f"调整 {m_name} 掩码大小失败: {e_resize}. 跳过此掩码。")
            else:
                # print(f"Warning: {m_name} mask is empty or invalid, skipping.")
                pass


        # 按优先级合并 (水体 > 植被 > 建设用地)
        # 应该从优先级最低的开始赋值，以确保高优先级区域覆盖低优先级的
        if 'builtup' in valid_masks: final_classification_map[valid_masks['builtup'] == 1] = 3
        if 'veg' in valid_masks: final_classification_map[valid_masks['veg'] == 1] = 1
        if 'water' in valid_masks: final_classification_map[valid_masks['water'] == 1] = 2


        # 提取裸地（从未分类的区域中提取）
        bareland_mask = extract_bareland_by_rule(features,
                                                 vegetation_mask=(final_classification_map == 1),
                                                 water_mask=(final_classification_map == 2),
                                                 builtup_mask=(final_classification_map == 3),
                                                 min_area=int(img_shape[0]*img_shape[1]*0.0005))
        if bareland_mask is not None and bareland_mask.shape == img_shape and bareland_mask.any():
            # Apply bareland mask only to areas not yet classified (label 0)
            final_classification_map[(bareland_mask == 1) & (final_classification_map == 0)] = 4
        title = "基于规则的地物分类结果"

    elif method == 'kmeans':
        print("\n--- K-Means 无监督分类 ---")
        # KMeans output labels start from 0. Our class IDs start from 1. Map accordingly.
        # Feature keys can be adjusted as needed, None means automatic selection
        kmeans_feature_keys = ['ndvi', 'ndwi', 'ndbi', 'texture_mean', 'hierarchical_all'] # Example, can add more or use hierarchical_all
        # Filter out keys not present in features or not valid arrays
        valid_kmeans_keys = [k for k in kmeans_feature_keys if k in features and isinstance(features[k], np.ndarray) and (features[k].ndim == 2 or features[k].ndim == 3)] # Allow 3D for hierarchical

        if not valid_kmeans_keys:
            print(f"警告: 为KMeans指定的特征键 {kmeans_feature_keys} 在数据中均不可用，将使用自动选择。")
            # Auto-selection is handled inside unsupervised_kmeans_classification now
            pass

        num_clusters_kmeans = 7 # Adjust number of clusters
        kmeans_result = unsupervised_kmeans_classification(features, n_clusters=num_clusters_kmeans, feature_keys_to_use=valid_kmeans_keys)

        # Map KMeans labels (0 to n-1) to class IDs (1 to n)
        final_classification_map = kmeans_result + 1

        # Update class names for KMeans results (Assuming n clusters map to IDs 1 to n)
        # User needs to interpret these clusters manually based on the image and features
        kmeans_class_names_map = { cid: f'KMeans 簇 {cid - np.min(kmeans_result) + 1}' for cid in np.unique(final_classification_map) if cid !=0 }
        class_names.update(kmeans_class_names_map)
        title = f"K-Means ({num_clusters_kmeans}簇) 无监督分类结果"


    elif method == 'random_forest':
        print("\n--- 随机森林 监督分类 ---")
        labeled_roi_file = "labeled_roi.tif" # <--- 用户需要提供此文件路径

        if not os.path.exists(labeled_roi_file):
            print(f"错误: 监督分类所需的标签ROI文件 '{labeled_roi_file}' 未找到。请创建此文件。")
            print("  该文件应为单波段栅格，像素值代表地物类别ID (0=忽略, 1=植被, 2=水体, ...)。")
            print("  其尺寸和地理配准应与输入影像一致。")
            return

        # Determine which features to use for Random Forest
        feature_array_for_rf = None
        feature_names_for_rf = []

        if use_hierarchical_all and 'hierarchical_all' in features and isinstance(features['hierarchical_all'], np.ndarray) and features['hierarchical_all'].ndim == 3 and features['hierarchical_all'].shape[:2] == img_shape:
            print("使用 'hierarchical_all' 特征数组进行随机森林分类。")
            feature_array_for_rf = features['hierarchical_all']
            # Generate placeholder names based on number of bands
            feature_names_for_rf = [f'hierarchical_feature_{i+1}' for i in range(feature_array_for_rf.shape[-1])]
        else:
            print("使用选定的单个特征进行随机森林分类（或'hierarchical_all'不可用/未请求）。")
            # Supervised classification feature keys (example)
            rf_feature_keys = ['ndvi', 'ndwi', 'ndbi', 'texture_mean', 'hierarchical_level_1'] # Can include hierarchical_level_1 or other features
            # Filter out keys not present or not valid 2D arrays matching image shape
            valid_rf_keys = [k for k in features.keys() if k in features and isinstance(features[k], np.ndarray) and features[k].ndim == 2 and features[k].shape == img_shape]

            if not valid_rf_keys:
                print(f"错误: 为随机森林指定的特征键 {rf_feature_keys} 在数据中均不可用或不是匹配图像形状的2D数组。")
                return

            # Stack selected 2D features into a single 3D array (H, W, n_features)
            feature_list_to_stack = [features[key] for key in valid_rf_keys]
            feature_array_for_rf = np.stack(feature_list_to_stack, axis=-1)
            feature_names_for_rf = valid_rf_keys

        if feature_array_for_rf is None or feature_array_for_rf.size == 0:
            print("错误: 未能准备特征数组用于随机森林训练和预测。")
            return
        if not feature_names_for_rf or len(feature_names_for_rf) != feature_array_for_rf.shape[-1]:
            print("错误: 特征名称列表为空或与数组中的特征数量不匹配。")
            # Generate placeholder names if missing
            feature_names_for_rf = [f'feature_{i+1}' for i in range(feature_array_for_rf.shape[-1])]
            print(f"使用生成的占位符特征名称: {feature_names_for_rf}")


        try:
            # Prepare training samples using the stacked feature array
            X_samples, y_labels = prepare_training_samples(feature_array_for_rf, labeled_roi_file)

            if X_samples.size == 0: # prepare_training_samples will raise error internally
                return

            print(f"训练样本数: {X_samples.shape[0]}")
            print(f"训练样本类别及数量: {dict(zip(*np.unique(y_labels, return_counts=True)))}")

            # Train or load classifier
            classifier_model_path = os.path.join(output_dir, "random_forest_model.joblib")
            if os.path.exists(classifier_model_path):
                print(f"加载已训练的随机森林模型: {classifier_model_path}")
                classifier = joblib.load(classifier_model_path)
                # Basic check to see if the loaded model matches the expected number of features
                if hasattr(classifier, 'n_features_in_') and classifier.n_features_in_ != feature_array_for_rf.shape[-1]:
                    print(f"警告: 加载的分类器需要 {classifier.n_features_in_} 个特征，但准备的数据有 {feature_array_for_rf.shape[-1]} 个。正在重新训练模型。")
                    classifier = train_random_forest_classifier(X_samples, y_labels, feature_names_for_training=feature_names_for_rf)
                    joblib.dump(classifier, classifier_model_path)
                    print(f"随机森林模型已重新训练并保存至: {classifier_model_path}")
                else:
                    print("加载的分类器与特征数据兼容。")

            else:
                classifier = train_random_forest_classifier(X_samples, y_labels, feature_names_for_training=feature_names_for_rf)
                joblib.dump(classifier, classifier_model_path)
                print(f"随机森林模型训练并保存至: {classifier_model_path}")

            # Predict using the trained classifier and the stacked feature array
            final_classification_map = supervised_classification_predict(feature_array_for_rf, classifier)
            title = "随机森林监督分类结果"

        except Exception as e_rf:
            print(f"随机森林分类过程中发生错误: {e_rf}")
            import traceback
            traceback.print_exc()
            return
    else:
        print(f"错误: 不支持的分割方法 '{method}'")
        return

    # 3. 可视化和保存结果
    if final_classification_map is not None and final_classification_map.size > 0:
        map_save_path = os.path.join(output_dir, f"{method}_classification_map.png")
        create_classification_map(final_classification_map, class_names, class_colors, save_path=map_save_path, title=title)

        # 保存为GeoTIFF (如果元数据完整)
        if all(k in features and features[k] is not None for k in ['transform', 'crs', 'width', 'height']):
            geotiff_save_path = os.path.join(output_dir, f"{method}_classification_map.tif")
            save_classification_as_geotiff(final_classification_map, features, geotiff_save_path)
        else:
            print("警告: 元数据不完整，无法将分类结果保存为带地理参考的GeoTIFF。")
            print(f"  需要: 'transform', 'crs', 'width', 'height'。可用: {list(features.keys())}")


    print("="*30)
    print("地物分割流程结束。")



def save_three_class_tif(class_map, meta, out_tif):
    H, W = class_map.shape
    # 读取 transform
    t = meta.get('transform') or meta.get('geo_transform')
    transform = t if isinstance(t, Affine) else Affine.from_gdal(*t)

    # 安全读取 CRS
    crs = None
    crs_meta = meta.get('crs')
    if crs_meta:
        crs = crs_meta if hasattr(crs_meta, 'to_wkt') else CRS.from_string(crs_meta)

    # 构造写入参数
    kwargs = {
        'driver': 'GTiff', 'height': H, 'width': W, 'count': 1,
        'dtype': rasterio.uint8, 'transform': transform, 'compress': 'lzw'
    }
    if crs:
        kwargs['crs'] = crs

    # 写 GeoTIFF
    with rasterio.open(out_tif, 'w', **kwargs) as dst:
        dst.write(class_map.astype(rasterio.uint8), 1)
        dst.write_colormap(1, {
            0: (0, 0, 0, 255),
            1: (0, 0, 255, 255),   # 1=水体(蓝)
            2: (0, 128, 0, 255),   # 2=植被(绿)
            3: (255, 0, 0, 255)    # 3=建设用地(红)
        })
    print(f"✅ 三类 GeoTIFF 已保存: {out_tif}")
    
    
    



if __name__ == "__main__":
    # --- 配置 ---
    # 设置您的特征提取后生成的 PKL 文件路径
    feature_file_to_test = "../output/feature_outputs/all_features_and_metadata.pkl"

    # 定义输出目录
    output_directory = "output/segmentation_results"

    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)

    # --- 创建虚拟特征文件用于演示 (如果指定的PKL文件不存在) ---
    # 如果您的PKL文件已存在，这部分代码不会执行
    if not os.path.exists(feature_file_to_test):
        print(f"警告: 特征文件 '{feature_file_to_test}' 不存在。")
        dummy_dir = os.path.dirname(feature_file_to_test)
        if not dummy_dir: dummy_dir = "modules/feature_outputs"
        os.makedirs(dummy_dir, exist_ok=True)

        print(f"将尝试在 '{dummy_dir}' 创建一个虚拟的 '{os.path.basename(feature_file_to_test)}' (PKL格式) 用于演示。")
        feature_file_to_test = os.path.join(dummy_dir, os.path.basename(feature_file_to_test))

        height, width = 256, 256
        # 模拟 pkl 文件内容结构，包含单个特征字典和元数据
        dummy_features_data_individual = {
            # 模拟一些水体、植被、土壤/建筑相关的指数
            'ndvi': np.random.rand(height, width) * 2 - 1,
            'ndwi': np.random.rand(height, width) * 2 - 1,
            'mndwi': np.random.rand(height, width) * 2 - 1, # Added MNDWI
            'ndbi': np.random.rand(height, width) * 2 - 1,
            'bsi': np.random.rand(height, width) * 2 - 1, # Added BSI
            'evi': np.random.rand(height, width) * 2 - 1, # Added EVI
            'texture_mean': np.random.rand(height, width) * 255,
            'pca_result_0': np.random.rand(height, width), # Example PCA component
            'hierarchical_level_1': np.random.rand(height, width, 5), # Example hierarchical level 1
            'hierarchical_level_2': np.random.rand(height, width, 3), # Example hierarchical level 2
            'hierarchical_all': np.random.rand(height, width, 8) # Example combined hierarchical
        }
        # 注意：为了模拟您的特征提取代码的输出结构，将单个特征放在 'all_extracted_features_dict' 下
        # 并将层次化特征放在 'hierarchical_features' 下
        dummy_pkl_content = {
            'all_extracted_features_dict': dummy_features_data_individual,
            'hierarchical_features': {
                'level_1': dummy_features_data_individual['hierarchical_level_1'],
                'level_2': dummy_features_data_individual['hierarchical_level_2'],
                'all': dummy_features_data_individual['hierarchical_all']
            },
            'dimensions': (height, width),
            'geo_transform': Affine.from_gdal(600000.0, 0.5, 0.0, 5400000.0, 0.0, -0.5).to_gdal(),
            'crs': rasterio.crs.CRS.from_epsg(32630).to_wkt()
        }
        try:
            with open(feature_file_to_test, 'wb') as f_dummy:
                pickle.dump(dummy_pkl_content, f_dummy)
            print(f"已成功创建虚拟特征文件: {feature_file_to_test}")
        except Exception as e_create_dummy:
            print(f"创建虚拟特征文件失败: {e_create_dummy}. 请确保您有权限写入该目录。")
            # exit() # Do not exit, allow trying with existing file or handle error


    # --- 执行可视化组合特征的功能 ---
    # 加载特征文件
    try:
        loaded_features = load_features(feature_file_to_test)
        # 规范化特征结构，确保指数在顶层可访问
        normalized_features = normalize_features_structure(loaded_features)

        # 调用可视化函数，将水体、植被、土壤相关特征组合可视化
        visualize_combined_indices(normalized_features, output_dir=output_directory, save_path="combined_indices_visualization.png")

        # 这里示例用 K-Means，无监督分类；你也可以换成 'rule_based' 或 'random_forest'
        run_classification_stage(
            feature_file_to_test,
            method='kmeans',
            output_dir=output_directory,
            use_hierarchical_all=True
        )

        # 显示集成指南
        integrate_three_class_output_to_main_workflow()
        
        
    except FileNotFoundError:
        print(f"错误: 指定的特征文件 '{feature_file_to_test}' 未找到。无法执行可视化。")
    except Exception as e:
        print(f"执行可视化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


    # --- 您也可以在这里继续调用分割工作流程 (任选一种方法) ---
    # 1. 基于规则的分类
    # main_segmentation_workflow(feature_file_to_test, method='rule_based', output_dir=output_directory)

    # 2. K-Means 无监督分类
    # main_segmentation_workflow(feature_file_to_test, method='kmeans', output_dir=output_directory)

    # 3. 随机森林 监督分类 (需要准备 labeled_roi.tif 文件)
    # 提示：运行随机森林前，请按以下说明准备 'labeled_roi.tif' 文件。
    #   - 文件名: labeled_roi.tif
    #   - 内容: 单波段栅格图像，与您的特征数据具有相同的尺寸和地理配准。
    #   - 像素值: 代表地物类别ID (例如, 0=忽略, 1=植被, 2=水体, ...)。
    #   - 创建工具: QGIS, ENVI, ArcGIS 等 GIS 软件。
    #
    # 如果您想测试随机森林流程，可以取消以下注释，并确保 `labeled_roi.tif` 文件存在于脚本同级目录。

    # labeled_roi_for_rf = "labeled_roi.tif"
    # if os.path.exists(labeled_roi_for_rf):
    #    print(f"\nRunning Random Forest classification using features from: {feature_file_to_test}")
    #    print(f"Using labeled ROI file: {labeled_roi_for_rf}")
    #    main_segmentation_workflow(feature_file_to_test, method='random_forest', output_dir=output_directory, use_hierarchical_all=True) # Set use_hierarchical_all as needed
    # else:
    #    print(f"\nSkipping Random Forest classification: Labeled ROI file '{labeled_roi_for_rf}' not found.")
    #    print("To run Random Forest, please create this file.")

    print("\n脚本执行完毕。")