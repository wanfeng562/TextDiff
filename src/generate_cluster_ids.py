# 根据提取的特征运行KMeans聚类，生成cluster_id映射文件
import os
import numpy as np
import json
from sklearn.cluster import MiniBatchKMeans
import argparse

def generate_cluster_ids(opts):
    # 加载提取的特征
    features = np.load(os.path.join(opts['exp_dir'], 'cluster_features.npy'))
    with open(os.path.join(opts['exp_dir'], 'filenames.json'), 'r') as f:
        filenames = json.load(f)
    
    # 运行KMeans聚类
    kmeans = MiniBatchKMeans(
        n_clusters=opts['num_clusters'],  # K值，从配置读取
        batch_size=1024,
        random_state=opts['seed'],
        n_init='auto'
    )
    cluster_ids = kmeans.fit_predict(features)  # [N,]
    
    # 保存聚类中心（可选，用于初始化嵌入层）
    np.save(os.path.join(opts['exp_dir'], 'cluster_centers.npy'), kmeans.cluster_centers_)
    
    # 生成文件名到cluster_id的映射
    cluster_id_map = {filename: int(cid) for filename, cid in zip(filenames, cluster_ids)}
    with open(os.path.join(opts['exp_dir'], 'cluster_id_map.json'), 'w') as f:
        json.dump(cluster_id_map, f)
    print(f"生成{opts['num_clusters']}个聚类，保存至{opts['exp_dir']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--num_clusters', type=int, default=12)  # K值
    args = parser.parse_args()
    opts = json.load(open(args.exp, 'r'))
    opts['num_clusters'] = args.num_clusters
    generate_cluster_ids(opts)