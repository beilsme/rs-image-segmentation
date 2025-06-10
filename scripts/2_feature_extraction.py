#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
'''
图像分割 - 特征提取与增强模块

功能：从预处理后的 TM 影像中提取各种光谱、纹理和空间特征，用于后续分类与分割

输入：预处理后的 TM 影像数据（各波段数组）
输出：特征字典（features_dict）、层次化特征数组（hierarchical_features）

作者：Meng Yinan
日期：2025-05-26
'''


import rasterio
import os
import pickle

from modules.features.indices import *

from modules.utils.set_chinese_font import set_chinese_font

set_chinese_font()


def run_feature_extraction_stage(bands_data, preprocessing=True, texture_band_index=3):
    """
    特征提取与增强主函数，优化后的层次化分类版本

    参数:
        bands_data: 原始波段数据列表，按TM波段顺序排列
        preprocessing: 是否需要进行预处理

    返回:
        features_dict: 包含各种特征的字典
        hierarchical_features: 为层次化分割准备的特征数组
    """
    # 波段数据预处理
    if preprocessing:
        print("进行数据预处理...")
        processed_bands = []
        for i, band in enumerate(bands_data):
            # 应用稳健归一化
            normalized_band = robust_normalize(band)
            processed_bands.append(normalized_band)
        bands_data = processed_bands

    # 提取各波段(假设bands_data包含7个TM波段，顺序为1-7)
    blue_band = bands_data[0]    # 波段1: 蓝光
    green_band = bands_data[1]   # 波段2: 绿光
    red_band = bands_data[2]     # 波段3: 红光
    nir_band = bands_data[3]     # 波段4: 近红外
    swir1_band = bands_data[4]   # 波段5: 短波红外1
    thermal_band = bands_data[5] if len(bands_data) > 5 else None  # 波段6: 热红外(可选)
    swir2_band = bands_data[6] if len(bands_data) > 6 else None    # 波段7: 短波红外2(可选)

    features_dict = {}

    # 1. 计算各种植被、水体和建筑指数
    print("计算特征指数...")
    # 植被指数
    features_dict['ndvi'] = calculate_ndvi(nir_band, red_band)
    features_dict['evi'] = calculate_evi(nir_band, red_band, blue_band)
    features_dict['msavi'] = calculate_msavi(nir_band, red_band)

    # 水体指数
    features_dict['ndwi'] = calculate_ndwi(green_band, nir_band)
    features_dict['mndwi'] = calculate_mndwi(green_band, swir1_band)

    # 建筑与裸土指数
    features_dict['ndbi'] = calculate_ndbi(swir1_band, nir_band)
    features_dict['bsi'] = calculate_bsi(blue_band, red_band, nir_band, swir1_band)

    # 2. 执行主成分分析
    print("执行主成分分析...")
    valid_bands = [band for band in bands_data if band is not None]
    pca_result, variance_ratio, pca_model = perform_pca(valid_bands, use_robust_scaling=True)
    features_dict['pca_result'] = pca_result
    features_dict['variance_ratio'] = variance_ratio

    # 3. 计算多种纹理特征
    print("计算纹理特征...")
    texture_band = nir_band

    # GLCM纹理特征
    print("  计算GLCM纹理特征...")
    glcm_features = calculate_glcm_features(texture_band)
    features_dict['glcm_features'] = glcm_features

    # LBP纹理特征
    print("  计算LBP纹理特征...")
    lbp_feature = calculate_lbp_features(texture_band)
    features_dict['lbp_feature'] = lbp_feature

    # 计算多尺度特征、形态学特征和滤波器响应
    print("计算多尺度特征...")
    multi_scale_features = calculate_multi_scale_features(texture_band)
    features_dict['multi_scale_features'] = multi_scale_features

    print("计算形态学特征...")
    morphological_features = calculate_morphological_features(texture_band)
    features_dict['morphological_features'] = morphological_features

    print("计算滤波器响应...")
    filter_features = calculate_filter_responses(texture_band)
    features_dict['filter_features'] = filter_features

    # 4. 新增: 创建层次化特征集 - 这是关键修改部分
    print("创建层次化特征集...")
    # 第一级特征 - 主要类别区分特征
    level_1_features = prepare_level_1_features(features_dict)

    # 第二级特征 - 细分类特征
    level_2_features = prepare_level_2_features(features_dict)

    # 5. 添加空间上下文信息
    print("添加空间上下文信息...")
    level_1_with_context = add_spatial_context(level_1_features)

    # 6. 准备最终的层次化分割特征
    print("准备层次化分割特征...")
    hierarchical_features = {
        'level_1': level_1_with_context,  # 用于主要类别区分的特征
        'level_2': level_2_features,      # 用于细分类的特征
        'all': np.concatenate([level_1_with_context, level_2_features], axis=-1)  # 完整特征集
    }

    # 7. 可视化部分重要特征
    print("可视化特征...")
    visualize_hierarchical_features(hierarchical_features, features_dict)

    return features_dict, hierarchical_features



if __name__ == "__main__":
    # 文件路径配置
    image_path = '../data/TM_image_AA_preprocessed.png/TM_image_AA_preprocessed.tif'  # 确保这是你的输入影像路径
    output_dir = '../output/feature_outputs'

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # === 数据加载 ===
    print(f"正在加载遥感图像数据: {image_path}...")
    bands_data = []
    transform = None
    crs = None
    height, width = 0, 0

    try:
        with rasterio.open(image_path) as src:
            # 读取所有波段并转换为float32，处理可能的NoData值
            bands_data = []
            for i in range(1, src.count + 1):
                band = src.read(i).astype(np.float32)
                if src.nodata is not None:
                    band[band == src.nodata] = np.nan # 将NoData值转换为NaN
                bands_data.append(band)

            # 获取地理信息
            transform = src.transform
            crs = src.crs

            # 获取数据维度
            height, width = src.height, src.width

        print(f"成功加载数据，形状: ({height}, {width}), 波段数: {len(bands_data)}")

        # === 特征提取 ===
        print("\n开始执行改进的层次化特征提取流程...")

        # 调用 main 函数进行特征提取
        # 在 main 函数内部会进行稳健归一化 (preprocessing=True)
        # 默认使用第4波段 (索引3) 计算纹理
        features_dict, hierarchical_features = run_feature_extraction_stage(bands_data, preprocessing=True, texture_band_index=3)

        # 检查特征提取是否成功生成了特征
        if not hierarchical_features or hierarchical_features['all'].shape[2] == 0:
            print("\n特征提取失败，没有生成用于分割的特征。程序终止。")
        else:
            print("\n特征提取完成，层次化特征集已准备就绪。")
            print(f"  第一级特征形状(主要类别): {hierarchical_features['level_1'].shape}")
            print(f"  第二级特征形状(细分类): {hierarchical_features['level_2'].shape}")
            print(f"  完整特征形状(L1+L2): {hierarchical_features['all'].shape}")


            # === 特征保存 ===
            print("\n保存提取的层次化特征...")
            # 1. NumPy二进制格式 (.npy)
            np_output_path_level1 = os.path.join(output_dir, 'level1_features.npy')
            np_output_path_level2 = os.path.join(output_dir, 'level2_features.npy')
            np_output_path_all = os.path.join(output_dir, 'all_hierarchical_features.npy')

            try:
                if hierarchical_features['level_1'].shape[2] > 0:
                    np.save(np_output_path_level1, hierarchical_features['level_1'])
                    print(f"- 第一级特征 (NumPy): {np_output_path_level1}")
                else:
                    print("- 没有第一级特征可保存为NumPy。")

                if hierarchical_features['level_2'].shape[2] > 0:
                    np.save(np_output_path_level2, hierarchical_features['level_2'])
                    print(f"- 第二级特征 (NumPy): {np_output_path_level2}")
                else:
                    print("- 没有第二级特征可保存为NumPy。")

                if hierarchical_features['all'].shape[2] > 0:
                    np.save(np_output_path_all, hierarchical_features['all'])
                    print(f"- 完整层次特征 (NumPy): {np_output_path_all}")
                else:
                    print("- 没有完整层次特征可保存为NumPy。")

            except Exception as e:
                print(f"保存 NumPy 文件时出错: {e}")


            # 2. Pickle格式 - 保留所有特征字典和元数据
            # 这个文件包含了所有详细的中间特征和最终的层次化特征，方便后续调试和分析
            feature_data_full = {
                'hierarchical_features': hierarchical_features, # 最终的层次化特征字典
                'all_extracted_features_dict': features_dict, # 所有提取的详细特征字典
                'dimensions': (height, width),
                'geo_transform': transform,
                'crs': crs
            }
            pickle_output_path = os.path.join(output_dir, 'all_features_and_metadata.pkl')
            try:
                with open(pickle_output_path, 'wb') as f:
                    pickle.dump(feature_data_full, f)
                print(f"- 所有特征和元数据 (Pickle): {pickle_output_path}")
            except Exception as e:
                print(f"保存 Pickle 文件时出错: {e}")


            # 3. 导出完整层次特征为GeoTIFF格式，保留地理参考信息
            if hierarchical_features['all'].shape[2] > 0:
                geotiff_output_path = os.path.join(output_dir, 'all_hierarchical_features.tif')
                print(f"保存完整层次特征为 GeoTIFF: {geotiff_output_path}...")
                try:
                    with rasterio.open(
                            geotiff_output_path,
                            'w',
                            driver='GTiff',
                            height=height,
                            width=width,
                            count=hierarchical_features['all'].shape[2], # 特征数量作为波段数
                            dtype=hierarchical_features['all'].dtype, # 使用特征的数据类型
                            crs=crs,
                            transform=transform,
                            compress='lzw', # 添加Lzw压缩
                            tiled=True, blockxsize=256, blockysize=256 # 分块存储
                    ) as dst:
                        for i in range(hierarchical_features['all'].shape[2]):
                            dst.write(hierarchical_features['all'][:, :, i], i + 1)
                    print(f"- 完整层次特征 (GeoTIFF): {geotiff_output_path}")
                except Exception as e:
                    print(f"保存 GeoTIFF 文件时出错: {e}")
            else:
                print("- 没有完整层次特征可保存为GeoTIFF。")

            print("特征保存完成。")


            # === 可视化增强 ===
            print("\n生成可视化图表...")

            # 1. 显示原始波段的假彩色合成图 (如果原始波段在 features_dict 中)
            if 'processed_bands' in features_dict and len(features_dict['processed_bands']) >= 4:
                print("  生成假彩色合成图...")
                plt.figure(figsize=(12, 10))
                # 使用近红外(3), 红(2), 绿(1)波段组合 (索引从0开始)
                # 确保波段存在且非None
                nir_vis = features_dict['processed_bands'][3] if len(features_dict['processed_bands']) > 3 and features_dict['processed_bands'][3] is not None else np.zeros((height, width))
                red_vis = features_dict['processed_bands'][2] if len(features_dict['processed_bands']) > 2 and features_dict['processed_bands'][2] is not None else np.zeros((height, width))
                green_vis = features_dict['processed_bands'][1] if len(features_dict['processed_bands']) > 1 and features_dict['processed_bands'][1] is not None else np.zeros((height, width))

                # 堆叠为RGB图像，确保范围在[0,1]
                rgb = np.dstack((nir_vis, red_vis, green_vis))
                rgb = np.clip(rgb, 0, 1)

                plt.imshow(rgb)
                plt.title('假彩色合成图 (近红外-红-绿)')
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, 'false_color_composite.png'), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                print("  警告: 原始波段不足或缺失，无法生成假彩色合成图。")

            # 2. 生成重要光谱指数的图像
            print("  生成重要光谱指数图...")
            indices_to_plot = {
                '归一化植被指数 (NDVI)': features_dict.get('ndvi'),
                '归一化水体指数 (NDWI)': features_dict.get('ndwi'),
                '改进的归一化水体指数 (MNDWI)': features_dict.get('mndwi'),
                '归一化建筑指数 (NDBI)': features_dict.get('ndbi'),
                '裸土指数 (BSI)': features_dict.get('bsi')
            }
            available_indices = {k: v for k, v in indices_to_plot.items() if v is not None}

            if available_indices:
                n_indices = len(available_indices)
                n_cols = min(3, n_indices)
                n_rows = (n_indices + n_cols - 1) // n_cols

                plt.figure(figsize=(6 * n_cols, 5 * n_rows))

                for i, (title, index_data) in enumerate(available_indices.items()):
                    plt.subplot(n_rows, n_cols, i + 1)
                    cmap = 'RdYlGn' if '植被' in title else ('Blues' if '水体' in title else ('Reds' if '建筑' in title else ('copper' if '裸土' in title else 'viridis')))
                    vmin = -1 if any(t in title for t in ['植被', '水体', '建筑', '裸土']) else np.min(index_data)
                    vmax = 1 if any(t in title for t in ['植被', '水体', '建筑', '裸土']) else np.max(index_data)

                    plt.imshow(index_data, cmap=cmap, vmin=vmin, vmax=vmax)
                    plt.colorbar(label='值', fraction=0.046, pad=0.04) # 简化colorbar标签
                    plt.title(title)
                    plt.axis('off')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(os.path.join(output_dir, 'spectral_indices.png'), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                print("  警告: 没有计算出可用的光谱指数进行可视化。")

            # 3. 可视化特征空间的主要成分 (PCA)
            if 'pca_result' in features_dict and features_dict['pca_result'] and len(features_dict['pca_result']) >= 3:
                print("  生成PCA可视化图...")
                pca_rgb = np.dstack((features_dict['pca_result'][0],
                                     features_dict['pca_result'][1],
                                     features_dict['pca_result'][2]))

                pca_normalized = np.zeros_like(pca_rgb, dtype=np.float32)
                for i in range(pca_rgb.shape[2]):
                    comp = pca_rgb[:,:,i]
                    min_val = np.min(comp)
                    max_val = np.max(comp)
                    if max_val > min_val:
                        pca_normalized[:,:,i] = (comp - min_val) / (max_val - min_val + 1e-10)
                    else:
                        pca_normalized[:,:,i] = np.zeros_like(comp)
                pca_normalized = np.clip(pca_normalized, 0, 1)

                plt.figure(figsize=(12, 10))
                plt.imshow(pca_normalized)
                plt.title('特征空间PCA可视化 (前三个主成分)')
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, 'feature_pca.png'), dpi=300, bbox_inches='tight')
                plt.close()

                if 'variance_ratio' in features_dict and features_dict['variance_ratio'].size > 0:
                    explained_variance = features_dict['variance_ratio']
                    n_components_to_plot = min(len(explained_variance), 10)
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(1, n_components_to_plot + 1), explained_variance[:n_components_to_plot])
                    plt.xlabel('主成分序号')
                    plt.ylabel('解释方差比例')
                    plt.title('PCA方差解释率')
                    plt.xticks(range(1, n_components_to_plot + 1))
                    plt.grid(axis='y', linestyle='--')
                    plt.savefig(os.path.join(output_dir, 'pca_variance_explained.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"  前 {n_components_to_plot} 个主成分解释方差比例: {explained_variance[:n_components_to_plot]}")
            elif 'pca_result' in features_dict and features_dict['pca_result']:
                print(f"  警告: PCA结果少于3个分量 ({len(features_dict['pca_result'])}个)，无法进行RGB可视化。")
                if 'variance_ratio' in features_dict and features_dict['variance_ratio'].size > 0:
                    explained_variance = features_dict['variance_ratio']
                    n_components_to_plot = min(len(explained_variance), 10)
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(1, n_components_to_plot + 1), explained_variance[:n_components_to_plot])
                    plt.xlabel('主成分序号')
                    plt.ylabel('解释方差比例')
                    plt.title('PCA方差解释率')
                    plt.xticks(range(1, n_components_to_plot + 1))
                    plt.grid(axis='y', linestyle='--')
                    plt.savefig(os.path.join(output_dir, 'pca_variance_explained.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"  前 {n_components_to_plot} 个主成分解释方差比例: {explained_variance[:n_components_to_plot]}")
            else:
                print("  警告: 没有计算出PCA结果。")

            # 4. 可视化层次化特征 (调用专门的可视化函数)
            # 这个函数会在 feature_outputs 目录下生成 level_1_features.png, level_2_features.png 等
            visualize_hierarchical_features(hierarchical_features, features_dict)


            # === 分割后处理的说明 ===
            print("\n=== 关于分割后处理的说明 ===")
            print("当前的脚本专注于特征提取。要实现地物分割和后处理（如合并河流湖泊），")
            print("你需要首先使用上述提取的层次化特征（特别是 'all' 数组）来训练和应用一个分类或分割模型。")
            print("在获得初步的像素级分类结果后，再进行后处理步骤。")
            print("\n例如，如果你有一个初步的分割结果数组 `segmentation_result` (每个像素一个类别ID)，")
            print("你可以使用 `semantic_merge_water_classes(segmentation_result)` 函数来合并水体类别。")
            print("更复杂的后处理可能需要结合第二级特征和空间信息来进行判断和合并。")
            print("这部分的实现取决于你选择的分类/分割算法和具体的后处理逻辑。")
            print("==========================")


            print(f"\n所有文件和可视化图表已保存在目录: {output_dir}")
            print("特征提取和准备流程完成。")

    except FileNotFoundError:
        print(f"错误: 未找到输入的图像文件: {image_path}")
    except Exception as e:
        print(f"处理图像时发生错误: {e}")
        import traceback
        traceback.print_exc() # 打印完整的错误堆栈，帮助调试
# 
# def main(bands_data, preprocessing=True, texture_band_index=3):
#     """
#     特征提取与增强主函数，优化后的层次化分类版本
#     
#     参数:
#         bands_data: 原始波段数据列表，按TM波段顺序排列
#         preprocessing: 是否需要进行预处理
#         
#     返回:
#         features_dict: 包含各种特征的字典
#         hierarchical_features: 为层次化分割准备的特征数组
#     """
#     # 波段数据预处理
#     if preprocessing:
#         print("进行数据预处理...")
#         processed_bands = []
#         for i, band in enumerate(bands_data):
#             # 应用稳健归一化
#             normalized_band = robust_normalize(band)
#             processed_bands.append(normalized_band)
#         bands_data = processed_bands
# 
#     # 提取各波段(假设bands_data包含7个TM波段，顺序为1-7)
#     blue_band = bands_data[0]    # 波段1: 蓝光
#     green_band = bands_data[1]   # 波段2: 绿光
#     red_band = bands_data[2]     # 波段3: 红光
#     nir_band = bands_data[3]     # 波段4: 近红外
#     swir1_band = bands_data[4]   # 波段5: 短波红外1
#     thermal_band = bands_data[5] if len(bands_data) > 5 else None  # 波段6: 热红外(可选)
#     swir2_band = bands_data[6] if len(bands_data) > 6 else None    # 波段7: 短波红外2(可选)
# 
#     features_dict = {}
# 
#     # 1. 计算各种植被、水体和建筑指数
#     print("计算特征指数...")
#     # 植被指数
#     features_dict['ndvi'] = calculate_ndvi(nir_band, red_band)
#     features_dict['evi'] = calculate_evi(nir_band, red_band, blue_band)
#     features_dict['msavi'] = calculate_msavi(nir_band, red_band)
# 
#     # 水体指数
#     features_dict['ndwi'] = calculate_ndwi(green_band, nir_band)
#     features_dict['mndwi'] = calculate_mndwi(green_band, swir1_band)
# 
#     # 建筑与裸土指数
#     features_dict['ndbi'] = calculate_ndbi(swir1_band, nir_band)
#     features_dict['bsi'] = calculate_bsi(blue_band, red_band, nir_band, swir1_band)
# 
#     # 2. 执行主成分分析
#     print("执行主成分分析...")
#     valid_bands = [band for band in bands_data if band is not None]
#     pca_result, variance_ratio, pca_model = perform_pca(valid_bands, use_robust_scaling=True)
#     features_dict['pca_result'] = pca_result
#     features_dict['variance_ratio'] = variance_ratio
# 
#     # 3. 计算多种纹理特征
#     print("计算纹理特征...")
#     texture_band = nir_band
# 
#     # GLCM纹理特征
#     print("  计算GLCM纹理特征...")
#     glcm_features = calculate_glcm_features(texture_band)
#     features_dict['glcm_features'] = glcm_features
# 
#     # LBP纹理特征
#     print("  计算LBP纹理特征...")
#     lbp_feature = calculate_lbp_features(texture_band)
#     features_dict['lbp_feature'] = lbp_feature
# 
#     # 计算多尺度特征、形态学特征和滤波器响应
#     print("计算多尺度特征...")
#     multi_scale_features = calculate_multi_scale_features(texture_band)
#     features_dict['multi_scale_features'] = multi_scale_features
# 
#     print("计算形态学特征...")
#     morphological_features = calculate_morphological_features(texture_band)
#     features_dict['morphological_features'] = morphological_features
# 
#     print("计算滤波器响应...")
#     filter_features = calculate_filter_responses(texture_band)
#     features_dict['filter_features'] = filter_features
# 
#     # 4. 新增: 创建层次化特征集 - 这是关键修改部分
#     print("创建层次化特征集...")
#     # 第一级特征 - 主要类别区分特征
#     level_1_features = prepare_level_1_features(features_dict)
# 
#     # 第二级特征 - 细分类特征
#     level_2_features = prepare_level_2_features(features_dict)
# 
#     # 5. 添加空间上下文信息
#     print("添加空间上下文信息...")
#     level_1_with_context = add_spatial_context(level_1_features)
# 
#     # 6. 准备最终的层次化分割特征
#     print("准备层次化分割特征...")
#     hierarchical_features = {
#         'level_1': level_1_with_context,  # 用于主要类别区分的特征
#         'level_2': level_2_features,      # 用于细分类的特征
#         'all': np.concatenate([level_1_with_context, level_2_features], axis=-1)  # 完整特征集
#     }
# 
#     # 7. 可视化部分重要特征
#     print("可视化特征...")
#     visualize_hierarchical_features(hierarchical_features, features_dict)
# 
#     return features_dict, hierarchical_features
# 
# 
# def run_feature_extraction_stage(input_file, output_dir, visualization_dir):
#     """
#     特征提取阶段主函数，与主控制器接口匹配
#     
#     参数:
#         input_file: 预处理后的输入影像文件路径
#         output_dir: 特征输出目录
#         visualization_dir: 可视化结果输出目录
#         
#     返回:
#         特征数据文件路径（成功时）或 None（失败时）
#     """
#     print(f"正在从 {input_file} 加载预处理后的遥感图像数据...")
# 
#     # 创建输出目录
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     if not os.path.exists(visualization_dir):
#         os.makedirs(visualization_dir)
# 
#     try:
#         # === 数据加载 ===
#         bands_data = []
#         transform = None
#         crs = None
#         height, width = 0, 0
# 
#         with rasterio.open(input_file) as src:
#             # 读取所有波段并转换为float32，处理可能的NoData值
#             bands_data = []
#             for i in range(1, src.count + 1):
#                 band = src.read(i).astype(np.float32)
#                 if src.nodata is not None:
#                     band[band == src.nodata] = np.nan # 将NoData值转换为NaN
#                 bands_data.append(band)
# 
#             # 获取地理信息
#             transform = src.transform
#             crs = src.crs
# 
#             # 获取数据维度
#             height, width = src.height, src.width
# 
#         print(f"成功加载数据，形状: ({height}, {width}), 波段数: {len(bands_data)}")
# 
#         # === 特征提取 ===
#         print("\n开始执行改进的层次化特征提取流程...")
# 
#         # 调用 main 函数进行特征提取
#         # 在 main 函数内部会进行稳健归一化 (preprocessing=True)
#         # 默认使用第4波段 (索引3) 计算纹理
#         features_dict, hierarchical_features = main(bands_data, preprocessing=True, texture_band_index=3)
# 
#         # 检查特征提取是否成功生成了特征
#         if not hierarchical_features or hierarchical_features['all'].shape[2] == 0:
#             print("\n特征提取失败，没有生成用于分割的特征。")
#             return None
#         else:
#             print("\n特征提取完成，层次化特征集已准备就绪。")
#             print(f"  第一级特征形状(主要类别): {hierarchical_features['level_1'].shape}")
#             print(f"  第二级特征形状(细分类): {hierarchical_features['level_2'].shape}")
#             print(f"  完整特征形状(L1+L2): {hierarchical_features['all'].shape}")
# 
#             # === 特征保存 ===
#             print("\n保存提取的层次化特征...")
#             # 1. NumPy二进制格式 (.npy)
#             np_output_path_level1 = os.path.join(output_dir, 'level1_features.npy')
#             np_output_path_level2 = os.path.join(output_dir, 'level2_features.npy')
#             np_output_path_all = os.path.join(output_dir, 'all_hierarchical_features.npy')
# 
#             try:
#                 if hierarchical_features['level_1'].shape[2] > 0:
#                     np.save(np_output_path_level1, hierarchical_features['level_1'])
#                     print(f"- 第一级特征 (NumPy): {np_output_path_level1}")
#                 else:
#                     print("- 没有第一级特征可保存为NumPy。")
# 
#                 if hierarchical_features['level_2'].shape[2] > 0:
#                     np.save(np_output_path_level2, hierarchical_features['level_2'])
#                     print(f"- 第二级特征 (NumPy): {np_output_path_level2}")
#                 else:
#                     print("- 没有第二级特征可保存为NumPy。")
# 
#                 if hierarchical_features['all'].shape[2] > 0:
#                     np.save(np_output_path_all, hierarchical_features['all'])
#                     print(f"- 完整层次特征 (NumPy): {np_output_path_all}")
#                 else:
#                     print("- 没有完整层次特征可保存为NumPy。")
# 
#             except Exception as e:
#                 print(f"保存 NumPy 文件时出错: {e}")
# 
#             # 2. Pickle格式 - 保留所有特征字典和元数据
#             # 这个文件包含了所有详细的中间特征和最终的层次化特征，方便后续调试和分析
#             feature_data_full = {
#                 'hierarchical_features': hierarchical_features, # 最终的层次化特征字典
#                 # 平铺四个归一化指数，供后续分类直接读取
#                 'ndvi':  features_dict.get('ndvi'),
#                 'ndwi':  features_dict.get('ndwi'),
#                 'mndwi': features_dict.get('mndwi'),
#                 'ndbi':  features_dict.get('ndbi'),
# 
#                 'all_extracted_features_dict': features_dict, # 所有提取的详细特征字典
#                 'dimensions': (height, width),
#                 'geo_transform': transform,
#                 'crs': crs
#             }
#             pickle_output_path = os.path.join(output_dir, 'all_features_and_metadata.pkl')
#             try:
#                 with open(pickle_output_path, 'wb') as f:
#                     pickle.dump(feature_data_full, f)
#                 print(f"- 所有特征和元数据 (Pickle): {pickle_output_path}")
#             except Exception as e:
#                 print(f"保存 Pickle 文件时出错: {e}")
# 
#             # 3. 导出完整层次特征为GeoTIFF格式，保留地理参考信息
#             if hierarchical_features['all'].shape[2] > 0:
#                 geotiff_output_path = os.path.join(output_dir, 'all_hierarchical_features.tif')
#                 print(f"保存完整层次特征为 GeoTIFF: {geotiff_output_path}...")
#                 try:
#                     with rasterio.open(
#                             geotiff_output_path,
#                             'w',
#                             driver='GTiff',
#                             height=height,
#                             width=width,
#                             count=hierarchical_features['all'].shape[2], # 特征数量作为波段数
#                             dtype=hierarchical_features['all'].dtype, # 使用特征的数据类型
#                             crs=crs,
#                             transform=transform,
#                             compress='lzw', # 添加Lzw压缩
#                             tiled=True, blockxsize=256, blockysize=256 # 分块存储
#                     ) as dst:
#                         for i in range(hierarchical_features['all'].shape[2]):
#                             dst.write(hierarchical_features['all'][:, :, i], i + 1)
#                     print(f"- 完整层次特征 (GeoTIFF): {geotiff_output_path}")
#                 except Exception as e:
#                     print(f"保存 GeoTIFF 文件时出错: {e}")
#             else:
#                 print("- 没有完整层次特征可保存为GeoTIFF。")
# 
#             print("特征保存完成。")
# 
#             # === 可视化增强 ===
#             print("\n生成可视化图表...")
# 
#             # 1. 显示原始波段的假彩色合成图 (如果原始波段在 features_dict 中)
#             if 'processed_bands' in features_dict and len(features_dict['processed_bands']) >= 4:
#                 print("  生成假彩色合成图...")
#                 plt.figure(figsize=(12, 10))
#                 # 使用近红外(3), 红(2), 绿(1)波段组合 (索引从0开始)
#                 # 确保波段存在且非None
#                 nir_vis = features_dict['processed_bands'][3] if len(features_dict['processed_bands']) > 3 and features_dict['processed_bands'][3] is not None else np.zeros((height, width))
#                 red_vis = features_dict['processed_bands'][2] if len(features_dict['processed_bands']) > 2 and features_dict['processed_bands'][2] is not None else np.zeros((height, width))
#                 green_vis = features_dict['processed_bands'][1] if len(features_dict['processed_bands']) > 1 and features_dict['processed_bands'][1] is not None else np.zeros((height, width))
# 
#                 # 堆叠为RGB图像，确保范围在[0,1]
#                 rgb = np.dstack((nir_vis, red_vis, green_vis))
#                 rgb = np.clip(rgb, 0, 1)
# 
#                 plt.imshow(rgb)
#                 plt.title('假彩色合成图 (近红外-红-绿)')
#                 plt.axis('off')
#                 plt.savefig(os.path.join(visualization_dir, 'false_color_composite.png'), dpi=300, bbox_inches='tight')
#                 plt.close()
#             else:
#                 print("  警告: 原始波段不足或缺失，无法生成假彩色合成图。")
# 
#             # 2. 生成重要光谱指数的图像
#             print("  生成重要光谱指数图...")
#             indices_to_plot = {
#                 '归一化植被指数 (NDVI)': features_dict.get('ndvi'),
#                 '归一化水体指数 (NDWI)': features_dict.get('ndwi'),
#                 '改进的归一化水体指数 (MNDWI)': features_dict.get('mndwi'),
#                 '归一化建筑指数 (NDBI)': features_dict.get('ndbi'),
#                 '裸土指数 (BSI)': features_dict.get('bsi')
#             }
#             available_indices = {k: v for k, v in indices_to_plot.items() if v is not None}
# 
#             if available_indices:
#                 n_indices = len(available_indices)
#                 n_cols = min(3, n_indices)
#                 n_rows = (n_indices + n_cols - 1) // n_cols
# 
#                 plt.figure(figsize=(6 * n_cols, 5 * n_rows))
# 
#                 for i, (title, index_data) in enumerate(available_indices.items()):
#                     plt.subplot(n_rows, n_cols, i + 1)
#                     cmap = 'RdYlGn' if '植被' in title else ('Blues' if '水体' in title else ('Reds' if '建筑' in title else ('copper' if '裸土' in title else 'viridis')))
#                     vmin = -1 if any(t in title for t in ['植被', '水体', '建筑', '裸土']) else np.min(index_data)
#                     vmax = 1 if any(t in title for t in ['植被', '水体', '建筑', '裸土']) else np.max(index_data)
# 
#                     plt.imshow(index_data, cmap=cmap, vmin=vmin, vmax=vmax)
#                     plt.colorbar(label='值', fraction=0.046, pad=0.04) # 简化colorbar标签
#                     plt.title(title)
#                     plt.axis('off')
# 
#                 plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#                 plt.savefig(os.path.join(visualization_dir, 'spectral_indices.png'), dpi=300, bbox_inches='tight')
#                 plt.close()
#             else:
#                 print("  警告: 没有计算出可用的光谱指数进行可视化。")
# 
#             # 3. 可视化特征空间的主要成分 (PCA)
#             if 'pca_result' in features_dict and features_dict['pca_result'] and len(features_dict['pca_result']) >= 3:
#                 print("  生成PCA可视化图...")
#                 pca_rgb = np.dstack((features_dict['pca_result'][0],
#                                      features_dict['pca_result'][1],
#                                      features_dict['pca_result'][2]))
# 
#                 pca_normalized = np.zeros_like(pca_rgb, dtype=np.float32)
#                 for i in range(pca_rgb.shape[2]):
#                     comp = pca_rgb[:,:,i]
#                     min_val = np.min(comp)
#                     max_val = np.max(comp)
#                     if max_val > min_val:
#                         pca_normalized[:,:,i] = (comp - min_val) / (max_val - min_val + 1e-10)
#                     else:
#                         pca_normalized[:,:,i] = np.zeros_like(comp)
#                 pca_normalized = np.clip(pca_normalized, 0, 1)
# 
#                 plt.figure(figsize=(12, 10))
#                 plt.imshow(pca_normalized)
#                 plt.title('特征空间PCA可视化 (前三个主成分)')
#                 plt.axis('off')
#                 plt.savefig(os.path.join(visualization_dir, 'feature_pca.png'), dpi=300, bbox_inches='tight')
#                 plt.close()
# 
#                 if 'variance_ratio' in features_dict and features_dict['variance_ratio'].size > 0:
#                     explained_variance = features_dict['variance_ratio']
#                     n_components_to_plot = min(len(explained_variance), 10)
#                     plt.figure(figsize=(10, 6))
#                     plt.bar(range(1, n_components_to_plot + 1), explained_variance[:n_components_to_plot])
#                     plt.xlabel('主成分序号')
#                     plt.ylabel('解释方差比例')
#                     plt.title('PCA方差解释率')
#                     plt.xticks(range(1, n_components_to_plot + 1))
#                     plt.grid(axis='y', linestyle='--')
#                     plt.savefig(os.path.join(visualization_dir, 'pca_variance_explained.png'), dpi=300, bbox_inches='tight')
#                     plt.close()
#                     print(f"  前 {n_components_to_plot} 个主成分解释方差比例: {explained_variance[:n_components_to_plot]}")
#             elif 'pca_result' in features_dict and features_dict['pca_result']:
#                 print(f"  警告: PCA结果少于3个分量 ({len(features_dict['pca_result'])}个)，无法进行RGB可视化。")
#                 if 'variance_ratio' in features_dict and features_dict['variance_ratio'].size > 0:
#                     explained_variance = features_dict['variance_ratio']
#                     n_components_to_plot = min(len(explained_variance), 10)
#                     plt.figure(figsize=(10, 6))
#                     plt.bar(range(1, n_components_to_plot + 1), explained_variance[:n_components_to_plot])
#                     plt.xlabel('主成分序号')
#                     plt.ylabel('解释方差比例')
#                     plt.title('PCA方差解释率')
#                     plt.xticks(range(1, n_components_to_plot + 1))
#                     plt.grid(axis='y', linestyle='--')
#                     plt.savefig(os.path.join(visualization_dir, 'pca_variance_explained.png'), dpi=300, bbox_inches='tight')
#                     plt.close()
#                     print(f"  前 {n_components_to_plot} 个主成分解释方差比例: {explained_variance[:n_components_to_plot]}")
#             else:
#                 print("  警告: 没有计算出PCA结果。")
# 
#             # 4. 可视化层次化特征 (调用专门的可视化函数)
#             # 这个函数会在 visualization_dir 目录下生成 level_1_features.png, level_2_features.png 等
#             visualize_hierarchical_features(hierarchical_features, features_dict)
# 
#             print(f"\n所有文件和可视化图表已保存")
#             print(f"特征文件目录: {output_dir}")
#             print(f"可视化目录: {visualization_dir}")
# 
#             # 返回主要的特征数据文件路径
#             return pickle_output_path
# 
#     except FileNotFoundError:
#         print(f"错误: 未找到输入的图像文件: {input_file}")
#         return None
#     except Exception as e:
#         print(f"处理图像时发生错误: {e}")
#         import traceback
#         traceback.print_exc()
#         return None
# 
# 
# if __name__ == "__main__":
#     # 为了向后兼容，保留原有的独立运行方式
#     image_path = '../data/TM_image_AA_preprocessed.tif'
#     output_dir = '../output/feature_outputs'
#     visualization_dir = '../visualizations'
# 
#     result = run_feature_extraction_stage(image_path, output_dir, visualization_dir)
#     if result:
#         print("特征提取模块独立运行完成")
#     else:
#         print("特征提取模块独立运行失败")