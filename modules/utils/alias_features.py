#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
'''
图像分割 - 特征字典键别名修补模块

功能：为 all_features_and_metadata.pkl 文件中的特征字典添加或重命名键别名，生成新的带别名的 PKL 文件

输入：  
  - 原始特征数据文件（all_features_and_metadata.pkl）

输出：  
  - 带别名特征字典的新文件（all_features_and_metadata_aliased.pkl）

作者：Meng Yinan  
日期：2025-05-26
'''

import pickle
import os

# 原始 PKL
src = "../../output/feature_outputs/all_features_and_metadata.pkl"
# 打补丁后的 PKL
dst = "../../output/feature_outputs/all_features_and_metadata_aliased.pkl"

data = pickle.load(open(src, "rb"))

# 取出那个 dict
feat = data['all_extracted_features_dict']

# 把内部键名改一份
aliases = {
    'ndvi'         : 'ndvi',
    'ndwi'         : 'ndwi',
    'mndwi'        : 'mndwi',
    'ndbi'         : 'ndbi',
    'bsi'          : 'bsi',
    'evi'          : 'evi',
    'texture_mean' : 'texture_mean',
}
for new_key, suffix in aliases.items():
    old_key = f"all_extracted_features_dict_{suffix}"
    if old_key in feat:
        feat[new_key] = feat[old_key]

# 写回 data 并存成新文件
data['all_extracted_features_dict'] = feat
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "wb") as f:
    pickle.dump(data, f)

print(f"✅ 已生成别名特征文件：{dst}")