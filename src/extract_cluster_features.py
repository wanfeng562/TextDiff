# 从 Diffusion-UNet 提取全局特征用于后续聚类分析。
# 本文件提供一个函数 `extract_cluster_features(opts)`，接收配置字典 opts，
# 负责加载预训练模型、读取数据集、在指定层上提取特征并保存到磁盘。
import os
import torch
import numpy as np
import json
from tqdm import tqdm

# 从 guided_diffusion 的脚本工具中导入默认参数和模型创建方法（可用于加载预训练权重）
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)

# 导入项目内的特征提取器工厂函数，它会根据 opts 创建对应的 FeatureExtractor
from src.feature_extractors import create_feature_extractor, collect_features

# 导入自定义数据集加载器 Mixdataset（只需要图像部分用于提特征）
from src.load_dataset import Mixdataset, ValGenerator

# 读取训练/验证文本信息的工具函数
from utils import read_text

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

    # 将模型移动到设备并设为 eval。注意：后续应调用 feature_extractor 本身（wrapper），
    # 而不是直接调用内部 model.forward，因为 wrapper 会返回注册的激活或适当的输出格式。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        feature_extractor.model.to(device)
        feature_extractor.model.eval()
    except Exception:
        # 如果 feature_extractor 没有 model 属性，忽略（防御性处理）
        pass

    # 2) 加载训练集（只需要图像信息用于特征提取）
    # read_text: 读取文本信息（例如文件名/标签）的辅助函数
    train_text = read_text(os.path.join(opts["training_path"], "Train_text.xlsx"))
    # ValGenerator: 变换生成器，用于生成验证/评估阶段需要的图像变换（不做数据增强）
    train_tf = ValGenerator(output_size=[opts["image_size"], opts["image_size"]])
    # Mixdataset: 返回只包含图像（和文件名等元信息）的 dataset
    # 如果只需要图像特征，可以避免加载文本模型以节省时间/内存
    # 当 load_text=False 时，Mixdataset 会跳过 tokenizer 和 bert 的加载
    train_dataset = Mixdataset(
        dataset_path=opts["training_path"],
        row_text=train_text,
        joint_transform=train_tf,
        load_text=False,
    )  # 这里只使用图像数据

    # DataLoader: 按 batch 并行加载数据。
    # - shuffle=False 保持顺序（聚类时常希望与原始文件名顺序一致以便映射）
    # - num_workers=4 使用 4 个子进程并行读取（与 GPU 数量无关，但受 CPU/IO 限制）
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opts["batch_size"],
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
            img = sample["image"].to(device)

            # 触发前向计算：优先调用 feature_extractor(wrapper)，因为 wrapper 会返回
            # 注册好的激活（list/dict 视具体 extractor 而定）。直接调用内部 model.forward
            # 会返回模型输出张量，而不是我们需要的激活字典/列表，从而导致 indexing 错误。
            # 这里兼容多种返回格式：如果 wrapper 返回 list/tuple（activation list），
            # 我们用 collect_features 将其转换为尺度字典；如果已经是 dict 则直接使用。
            if callable(feature_extractor):
                # 有些 extractor 期望额外参数 like noise/timesteps; we try common signatures
                try:
                    activations_raw = feature_extractor(img)
                except TypeError:
                    # try with noise/timesteps arg (fallback)
                    try:
                        activations_raw = feature_extractor(
                            img,
                            timesteps=torch.zeros(img.size(0), dtype=torch.long).to(
                                device
                            ),
                        )
                    except Exception:
                        # last resort: call internal model.forward
                        activations_raw = feature_extractor.model.forward(img)
            else:
                activations_raw = feature_extractor.model.forward(img)

            # 将不同格式的 activations 规范为一个尺度->tensor 的字典
            if isinstance(activations_raw, dict):
                feats_dict = activations_raw
            elif isinstance(activations_raw, (list, tuple)):
                feats_dict = collect_features(activations_raw)
            elif isinstance(activations_raw, torch.Tensor):
                # 如果返回的是单个张量，尝试将其视为最细尺度（32x32）
                feats_dict = {"layer_32x32": activations_raw}
            else:
                raise RuntimeError(
                    f"Unsupported activations type: {type(activations_raw)}"
                )

            # 下游代码期望在 feats_dict 中有 'layer_32x32'
            if "layer_32x32" not in feats_dict:
                # 如果没有该尺度，尝试取第一个可用的尺度
                first_key = next(iter(feats_dict.keys()))
                print(
                    f"Warning: 'layer_32x32' not found in activations; using '{first_key}' instead."
                )
                feat = feats_dict[first_key]
            else:
                feat = feats_dict["layer_32x32"]

            # 对每个空间特征图做自适应平均池化到 1x1，然后 squeeze 成 [B, C]
            gap_feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1)).squeeze()

            # 将特征从 GPU 移到 CPU 并转为 numpy，追加到列表中
            features.append(gap_feat.cpu().numpy())

            # 把样本的文件名或标识追加到 filenames 列表中，保持与 features 的顺序一致
            filenames.extend(sample["name"])

    # 5) 将所有批次的特征按第 0 维拼接为最终的 [N, C] 数组并保存到磁盘
    features = np.concatenate(features, axis=0)
    np.save(os.path.join(opts["exp_dir"], "cluster_features.npy"), features)

    # 保存文件名映射，便于后续将聚类结果映射回原始样本
    with open(os.path.join(opts["exp_dir"], "filenames.json"), "w") as f:
        json.dump(filenames, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Merge guided_diffusion defaults like train.py does so that create_model_and_diffusion
    # receives all expected args (e.g., class_cond) even if not present in experiment json.
    from guided_diffusion.script_util import (
        model_and_diffusion_defaults,
        add_dict_to_argparser,
    )

    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    parser.add_argument("--exp", type=str, required=True)  # 配置文件路径
    parser.add_argument("--seed", type=int, default=40)
    args = parser.parse_args()
    opts = json.load(open(args.exp, "r"))
    # Values from CLI (including defaults added above) override experiment config
    opts.update(vars(args))
    os.makedirs(opts["exp_dir"], exist_ok=True)
    # ensure experiment save dir is unique (similar to train.py behavior)
    opts["exp_dir"] = os.path.join(
        opts["exp_dir"], f'experiment-{len(os.listdir(opts["exp_dir"]))+ 1:02d}'
    )
    os.makedirs(opts["exp_dir"], exist_ok=True)
    extract_cluster_features(opts)
