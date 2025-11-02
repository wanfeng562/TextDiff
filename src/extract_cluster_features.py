# 从 Diffusion-UNet 提取全局特征用于后续聚类分析。
# 本文件提供一个函数 `extract_cluster_features(opts)`，接收配置字典 opts，
# 负责加载预训练模型、读取数据集、在指定层上提取特征并保存到磁盘。
import os
import torch
import numpy as np
import json
from tqdm import tqdm

# 从 guided_diffusion 的脚本工具中导入默认参数和模型创建方法（可用于加载预训练权重）
from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion

# 导入项目内的特征提取器工厂函数，它会根据 opts 创建对应的 FeatureExtractor
from src.feature_extractors import create_feature_extractor

# 导入自定义数据集加载器 Mixdataset（只需要图像部分用于提特征）
from src.load_dataset import Mixdataset

# PyTorch 的 DataLoader，用于并行读取数据
from torch.utils.data import DataLoader
import argparse


def extract_cluster_features(opts):
    """
    逐句说明：
    - opts: 字典，包含模型路径、数据路径、提取哪几层（blocks）、图像大小、batch_size 等配置信息。

    函数流程概览：
    1. 用 create_feature_extractor 创建一个封装了预训练模型的特征提取器（FeatureExtractor）。
    2. 将模型设为 eval 模式并移动到合适设备（CUDA/CPU）。
    3. 构造数据集与 DataLoader，逐批读取图像。
    4. 对每个 batch 运行前向，读取注册在 decoder 层上的 hook 保存的激活（activations），
       对所需尺度做 GAP（全局平均池化）得到向量特征，并收集文件名。
    5. 将所有提取的特征合并并保存到 exp_dir 下，文件名映射也会保存以便后续匹配。
    """

    # 1) 初始化 Diffusion 模型的特征提取器（使用用户传入的 opts）
    # create_feature_extractor 会依据 opts['model_type'] 等参数返回相应的 FeatureExtractor 对象
    feature_extractor = create_feature_extractor(**opts)

    # 从封装器中取出实际的模型并设置为评估模式，避免 BatchNorm/Dropout 在推断时改变行为
    model = feature_extractor.model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 2) 加载训练集（只需要图像信息用于特征提取）
    # read_text: 读取文本信息（例如文件名/标签）的辅助函数
    train_text = read_text(os.path.join(opts['training_path'], 'Train_text.xlsx'))
    # ValGenerator: 变换生成器，用于生成验证/评估阶段需要的图像变换（不做数据增强）
    train_tf = ValGenerator(output_size=[opts['image_size'], opts['image_size']])
    # Mixdataset: 返回只包含图像（和文件名等元信息）的 dataset
    train_dataset = Mixdataset(
        dataset_path=opts['training_path'],
        row_text=train_text,
        joint_transform=train_tf,
    )  # 这里只使用图像数据

    # DataLoader: 按 batch 并行加载数据。
    # - shuffle=False 保持顺序（聚类时常希望与原始文件名顺序一致以便映射）
    # - num_workers=4 使用 4 个子进程并行读取（与 GPU 数量无关，但受 CPU/IO 限制）
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opts['batch_size'],
        shuffle=False,
        num_workers=4,
    )

    # 3) 用于收集特征和对应文件名的容器
    features = []  # 列表形式，每个元素是一个 [B, C] 的 numpy 数组（通过 GAP 得到）
    filenames = []  # 对应每个样本的文件名或 id，便于后续将聚类结果映射回原文件

    # 4) 提取特征：在 no_grad 模式下遍历数据并执行模型前向
    with torch.no_grad():
        # 遍历 DataLoader（tqdm 提供进度条，便于观察处理进度）
        for sample in tqdm(train_loader):
            # 从样本字典中取出图像并移动到目标设备（可能是 GPU）
            img = sample['image'].to(device)

            # 触发 UNet 的前向计算。注意：这里直接调用 model.forward，会运行模型并触发
            # 之前在 FeatureExtractorDDPM 中注册在指定 output_blocks 上的 forward hook，
            # 这些 hook 会把对应层的 activations 保存到相应的 block 上（或由 FeatureExtractor.forward 返回）。
            activations = feature_extractor.model.forward(
                img, timesteps=torch.zeros(img.size(0), dtype=torch.long).to(device)
            )

            # 下游代码假定 activations 是一个字典并包含特定尺度键 'layer_32x32'。
            # 这里演示如何从 activations 中选择特定尺度的特征并做全局平均池化（GAP），
            # 得到一个 [B, C] 的向量表示用于聚类。
            feat = activations['layer_32x32']  # 假设 activations 是 dict 并包含该 key

            # 对每个空间特征图做自适应平均池化到 1x1，然后 squeeze 成 [B, C]
            gap_feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1)).squeeze()

            # 将特征从 GPU 移到 CPU 并转为 numpy，追加到列表中
            features.append(gap_feat.cpu().numpy())

            # 把样本的文件名或标识追加到 filenames 列表中，保持与 features 的顺序一致
            filenames.extend(sample['name'])

    # 5) 将所有批次的特征按第 0 维拼接为最终的 [N, C] 数组并保存到磁盘
    features = np.concatenate(features, axis=0)
    np.save(os.path.join(opts['exp_dir'], 'cluster_features.npy'), features)

    # 保存文件名映射，便于后续将聚类结果映射回原始样本
    with open(os.path.join(opts['exp_dir'], 'filenames.json'), 'w') as f:
        json.dump(filenames, f)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True)  # 配置文件路径
    args = parser.parse_args()
    opts = json.load(open(args.exp, 'r'))
    os.makedirs(opts['exp_dir'], exist_ok=True)
    extract_cluster_features(opts)