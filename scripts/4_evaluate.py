#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
'''
图像分割 - 遥感影像分类结果评估模块

功能：将非监督聚类结果与人工采样点掩膜进行精度评估

输入：聚类结果特征文件、ROI掩膜文件
输出：混淆矩阵、精度评估报告、可视化对比图

作者：Meng Yinan
日期：2025-05-26
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
import pandas as pd
import os
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')
from modules.utils.set_chinese_font import set_chinese_font

set_chinese_font()

class ClassificationEvaluator:
    """分类结果评估器"""

    def __init__(self):
        # 定义类别映射（根据您的实际分类调整）
        self.class_mapping = {
            0: '未分类/背景',
            1: '植被',
            2: '水体',
            3: '建设用地',
            4: '裸地/其他'
        }

        # 定义颜色映射
        self.color_mapping = {
            0: [0, 0, 0],        # 黑色 - 未分类
            1: [0, 128, 0],      # 绿色 - 植被
            2: [0, 0, 255],      # 蓝色 - 水体
            3: [255, 0, 0],      # 红色 - 建设用地
            4: [255, 255, 0]     # 黄色 - 裸地
        }

    def load_classification_result(self, file_path):
        """加载分类结果文件"""
        if file_path.endswith('.npy'):
            return np.load(file_path)
        elif file_path.endswith('.tif') or file_path.endswith('.tiff'):
            import rasterio
            with rasterio.open(file_path) as src:
                return src.read(1)
        else:
            raise ValueError("不支持的文件格式，请使用 .npy 或 .tif 文件")

    def load_roi_mask(self, file_path):
        """加载ROI掩膜文件"""
        if file_path.endswith('.npy'):
            return np.load(file_path)
        elif file_path.endswith('.tif') or file_path.endswith('.tiff'):
            import rasterio
            with rasterio.open(file_path) as src:
                return src.read(1)
        else:
            raise ValueError("不支持的文件格式，请使用 .npy 或 .tif 文件")

    def extract_valid_samples(self, classification_map, roi_mask):
        """提取有效的采样点数据"""
        # 确保两个数组形状一致
        if classification_map.shape != roi_mask.shape:
            print(f"警告：分类图像形状 {classification_map.shape} 与ROI掩膜形状 {roi_mask.shape} 不一致")
            # 尝试调整大小
            from skimage.transform import resize
            roi_mask = resize(roi_mask, classification_map.shape,
                              order=0, preserve_range=True, anti_aliasing=False).astype(roi_mask.dtype)

        # 提取非零位置的样本（排除背景类别0）
        valid_mask = roi_mask > 0

        if not np.any(valid_mask):
            raise ValueError("ROI掩膜中没有找到有效的采样点")

        y_true = roi_mask[valid_mask]  # 真实标签
        y_pred = classification_map[valid_mask]  # 预测标签

        print(f"提取到 {len(y_true)} 个有效采样点")
        print(f"真实标签类别: {np.unique(y_true)}")
        print(f"预测标签类别: {np.unique(y_pred)}")

        return y_true, y_pred, valid_mask

    def map_clusters_to_classes(self, y_true, y_pred):
        """将聚类结果映射到真实类别"""
        # 获取唯一的聚类标签和真实类别
        unique_clusters = np.unique(y_pred)
        unique_classes = np.unique(y_true)

        print(f"\n聚类标签: {unique_clusters}")
        print(f"真实类别: {unique_classes}")

        # 创建映射字典
        cluster_to_class = {}

        # 对每个聚类，找到最频繁对应的真实类别
        for cluster in unique_clusters:
            cluster_mask = (y_pred == cluster)
            cluster_true_labels = y_true[cluster_mask]

            if len(cluster_true_labels) > 0:
                # 找到该聚类中最频繁的真实类别
                unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
                most_frequent_class = unique_labels[np.argmax(counts)]
                cluster_to_class[cluster] = most_frequent_class

                print(f"聚类 {cluster} -> 类别 {most_frequent_class} "
                      f"({self.class_mapping.get(most_frequent_class, '未知')})")

        # 应用映射
        y_pred_mapped = np.copy(y_pred)
        for cluster, class_label in cluster_to_class.items():
            y_pred_mapped[y_pred == cluster] = class_label

        return y_pred_mapped, cluster_to_class

    def calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        # 基本精度指标
        overall_accuracy = accuracy_score(y_true, y_pred)
        kappa_coefficient = cohen_kappa_score(y_true, y_pred)

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        # 分类报告
        class_names = [self.class_mapping.get(i, f'类别{i}') for i in np.unique(np.concatenate([y_true, y_pred]))]
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

        # 计算每类别的精度指标
        class_metrics = {}
        for class_name in class_names:
            if class_name in report:
                class_metrics[class_name] = {
                    'precision': report[class_name]['precision'],
                    'recall':    report[class_name]['recall'],
                    'f1-score':  report[class_name]['f1-score'],
                    'support':   report[class_name]['support']
                }

        return {
            'overall_accuracy': overall_accuracy,
            'kappa_coefficient': kappa_coefficient,
            'confusion_matrix': cm,
            'classification_report': report,
            'class_metrics': class_metrics
        }

    def plot_confusion_matrix(self, cm, class_names, save_path=None):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))

        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # 创建标注文本
        annotations = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

        # 绘制热力图
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': '样本数量'})

        plt.title('混淆矩阵', fontsize=16, fontweight='bold')
        plt.xlabel('预测类别', fontsize=12)
        plt.ylabel('真实类别', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵图已保存至: {save_path}")

        plt.show()

    def plot_accuracy_comparison(self, metrics, save_path=None):
        """绘制精度对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 总体精度指标
        overall_metrics = {
            '总体精度': metrics['overall_accuracy'] * 100,
            'Kappa系数': metrics['kappa_coefficient'] * 100
        }

        bars1 = ax1.bar(overall_metrics.keys(), overall_metrics.values(),
                        color=['skyblue', 'lightcoral'])
        ax1.set_title('总体精度指标', fontsize=14, fontweight='bold')
        ax1.set_ylabel('精度 (%)', fontsize=12)
        ax1.set_ylim(0, 100)

        # 添加数值标签
        for bar, value in zip(bars1, overall_metrics.values()):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')

        # 各类别精度指标
        if metrics['class_metrics']:
            class_names = list(metrics['class_metrics'].keys())
            precision_values = [metrics['class_metrics'][name]['precision'] * 100
                                for name in class_names]
            recall_values = [metrics['class_metrics'][name]['recall'] * 100
                             for name in class_names]
            f1_values = [metrics['class_metrics'][name]['f1-score'] * 100
                         for name in class_names]

            x = np.arange(len(class_names))
            width = 0.25

            bars2 = ax2.bar(x - width, precision_values, width, label='精确度', alpha=0.8)
            bars3 = ax2.bar(x, recall_values, width, label='召回率', alpha=0.8)
            bars4 = ax2.bar(x + width, f1_values, width, label='F1分数', alpha=0.8)

            ax2.set_title('各类别精度指标', fontsize=14, fontweight='bold')
            ax2.set_ylabel('精度 (%)', fontsize=12)
            ax2.set_xlabel('类别', fontsize=12)
            ax2.set_xticks(x)
            ax2.set_xticklabels(class_names, rotation=45, ha='right')
            # 将图例移到右下角，不遮挡柱子
            ax2.legend(loc='lower right', bbox_to_anchor=(1, 0), fontsize=10)
            ax2.set_ylim(0, 100)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"精度对比图已保存至: {save_path}")

        plt.show()

    def plot_classification_comparison(self, classification_map, roi_mask,
                                       valid_mask, save_path=None):
        """绘制分类结果对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 创建颜色映射
        def create_colored_image(label_map, color_map):
            colored = np.zeros((*label_map.shape, 3), dtype=np.uint8)
            for label, color in color_map.items():
                mask = (label_map == label)
                colored[mask] = color
            return colored

        # 分类结果图
        classification_colored = create_colored_image(classification_map, self.color_mapping)
        axes[0].imshow(classification_colored)
        axes[0].set_title('聚类分类结果', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # ROI掩膜图
        roi_colored = create_colored_image(roi_mask, self.color_mapping)
        axes[1].imshow(roi_colored)
        axes[1].set_title('人工采样点标签', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # 采样点位置图
        sample_display = np.zeros_like(classification_map)
        sample_display[valid_mask] = roi_mask[valid_mask]
        sample_colored = create_colored_image(sample_display, self.color_mapping)
        axes[2].imshow(sample_colored)
        axes[2].set_title('有效采样点分布', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        # 添加图例
        legend_elements = [Patch(facecolor=np.array(color)/255, label=label)
                           for label, color in self.color_mapping.items()
                           if label in np.unique(np.concatenate([classification_map.flatten(),
                                                                 roi_mask.flatten()]))]

        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02),
                   ncol=len(legend_elements), fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"分类对比图已保存至: {save_path}")

        plt.show()

    def generate_evaluation_report(self, metrics, cluster_mapping, output_path):
        """生成评估报告"""
        report_content = []
        report_content.append("="*60)
        report_content.append("遥感影像分类精度评估报告")
        report_content.append("="*60)
        report_content.append("")

        # 聚类映射信息
        report_content.append("聚类到类别的映射关系:")
        for cluster, class_label in cluster_mapping.items():
            class_name = self.class_mapping.get(class_label, f'类别{class_label}')
            report_content.append(f"  聚类 {cluster} -> {class_name}")
        report_content.append("")

        # 总体精度指标
        report_content.append("总体精度指标:")
        report_content.append(f"  总体精度: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
        report_content.append(f"  Kappa系数: {metrics['kappa_coefficient']:.4f}")
        report_content.append("")

        # 各类别精度指标
        report_content.append("各类别精度指标:")
        if metrics['class_metrics']:
            for class_name, class_metric in metrics['class_metrics'].items():
                report_content.append(f"  {class_name}:")
                report_content.append(f"    精确度: {class_metric['precision']:.4f} ({class_metric['precision']*100:.2f}%)")
                report_content.append(f"    召回率: {class_metric['recall']:.4f} ({class_metric['recall']*100:.2f}%)")
                report_content.append(f"    F1分数: {class_metric['f1-score']:.4f} ({class_metric['f1-score']*100:.2f}%)")
                report_content.append(f"    样本数: {class_metric['support']}")
                report_content.append("")

        # 混淆矩阵
        report_content.append("混淆矩阵:")
        cm = metrics['confusion_matrix']
        report_content.append("        " + "  ".join([f"{i:>8}" for i in range(len(cm))]))
        for i, row in enumerate(cm):
            report_content.append(f"  {i:>2}    " + "  ".join([f"{val:>8}" for val in row]))
        report_content.append("")

        # 保存报告
        report_text = "\n".join(report_content)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"评估报告已保存至: {output_path}")
        print("\n" + report_text)

    def evaluate_classification(self, classification_file, roi_mask_file, output_dir="evaluation_results"):
        """完整的分类评估流程"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        print("开始分类精度评估...")
        print("="*50)

        # 1. 加载数据
        print("1. 加载数据文件...")
        classification_map = self.load_classification_result(classification_file)
        roi_mask = self.load_roi_mask(roi_mask_file)

        print(f"分类结果形状: {classification_map.shape}")
        print(f"ROI掩膜形状: {roi_mask.shape}")

        # 2. 提取有效样本
        print("\n2. 提取有效采样点...")
        y_true, y_pred, valid_mask = self.extract_valid_samples(classification_map, roi_mask)

        # 3. 映射聚类结果到真实类别
        print("\n3. 映射聚类结果到真实类别...")
        y_pred_mapped, cluster_mapping = self.map_clusters_to_classes(y_true, y_pred)

        # 4. 计算评估指标
        print("\n4. 计算评估指标...")
        metrics = self.calculate_metrics(y_true, y_pred_mapped)

        # 5. 生成可视化结果
        print("\n5. 生成可视化结果...")

        # 获取类别名称
        unique_classes = np.unique(np.concatenate([y_true, y_pred_mapped]))
        class_names = [self.class_mapping.get(i, f'类别{i}') for i in unique_classes]

        # 绘制混淆矩阵
        confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
        self.plot_confusion_matrix(metrics['confusion_matrix'], class_names, confusion_matrix_path)

        # 绘制精度对比图
        accuracy_comparison_path = os.path.join(output_dir, "accuracy_comparison.png")
        self.plot_accuracy_comparison(metrics, accuracy_comparison_path)

        # 绘制分类对比图
        classification_comparison_path = os.path.join(output_dir, "classification_comparison.png")
        self.plot_classification_comparison(classification_map, roi_mask, valid_mask,
                                            classification_comparison_path)

        # 6. 生成评估报告
        print("\n6. 生成评估报告...")
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        self.generate_evaluation_report(metrics, cluster_mapping, report_path)

        print("\n" + "="*50)
        print("分类精度评估完成！")
        print(f"所有结果已保存至目录: {output_dir}")

        return metrics, cluster_mapping


def main():
    """主函数 - 使用示例"""

    # 创建评估器实例
    evaluator = ClassificationEvaluator()

    # 设置文件路径 - 请根据您的实际文件路径修改
    classification_result_file = "../output/feature_outputs/all_hierarchical_features.npy"  # 或 .npy
    roi_mask_file = "../output/ROI/roi_mask.npy"
    output_directory = "output/evaluation_results"

    # 检查文件是否存在
    if not os.path.exists(classification_result_file):
        print(f"错误: 分类结果文件不存在: {classification_result_file}")
        print("请确保已运行分类流程并生成了分类结果文件")
        return

    if not os.path.exists(roi_mask_file):
        print(f"错误: ROI掩膜文件不存在: {roi_mask_file}")
        print("请确保已生成ROI掩膜文件")
        return

    try:
        # 执行评估
        metrics, cluster_mapping = evaluator.evaluate_classification(
            classification_result_file,
            roi_mask_file,
            output_directory
        )

        print(f"\n评估完成! 总体精度: {metrics['overall_accuracy']*100:.2f}%")
        print(f"Kappa系数: {metrics['kappa_coefficient']:.4f}")

    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()