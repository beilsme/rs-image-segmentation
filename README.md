	•	Modules（用于交互式采样和监督分类前的准备）
	1.	sample_picker.py：在真彩色影像上点击采样，生成坐标和标签的 PKL 文件
	2.	roi.py：根据采样点 PKL 和参考影像生成 ROI 掩膜（.npy）
	3.	classifiers.py：加载特征、训练/预测（随机森林、KMeans、规则分类），输出分类地图
	•	Scripts（按顺序自动化整个流程）
	1.	1_preprocessing.py：TM 影像的辐射校正、几何校正和增强，输出预处理结果
	2.	2_feature_extraction.py：从预处理影像提取各种光谱/纹理/PCA 等特征，保存为 .npy/.pkl/.tif
	3.	3_classification.py：基于提取的特征进行地物分类，支持规则、KMeans 和随机森林；生成彩色分类图和 GeoTIFF
	4.	4_accuracy.py：读取分类结果和 ROI 掩膜，计算混淆矩阵、OA、Kappa，并输出可视化图和文本报告