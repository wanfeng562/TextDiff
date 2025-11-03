# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os

import cv2
from scipy import ndimage
from transformers import AutoTokenizer, AutoModel
import natsort
import json


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text, name = (
            sample["image"],
            sample["label"],
            sample["text"],
            sample["name"],
        )
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(
                image, (self.output_size[0] / x, self.output_size[1] / y), order=3
            )  # why not 3?
            label = zoom(
                label, (self.output_size[0] / x, self.output_size[1] / y), order=0
            )
        image = F.to_tensor(image)
        image = 2 * image - 1.0
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {"image": image, "label": label, "text": text, "name": name}
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text, name = (
            sample["image"],
            sample["label"],
            sample["text"],
            sample["name"],
        )
        image, label = image.astype(np.uint8), label.astype(np.uint8)  # OSIC
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(
                image, (self.output_size[0] / x, self.output_size[1] / y), order=3
            )
            label = zoom(
                label, (self.output_size[0] / x, self.output_size[1] / y), order=0
            )
        image = F.to_tensor(image)
        image = 2 * image - 1.0
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {"image": image, "label": label, "text": text, "name": name}
        return sample


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class LV2D(Dataset):
    def __init__(
        self,
        dataset_path: str,
        task_name: str,
        row_text: str,
        joint_transform: Callable = None,
        one_hot_mask: int = False,
        image_size: int = 224,
    ) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.output_path = os.path.join(dataset_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.bert_embedding = BertEmbedding()

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.output_path))

    def __getitem__(self, idx):

        mask_filename = self.mask_list[idx]  # Co
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1
        mask = correct_dims(mask)
        text = self.rowtext[mask_filename]
        text = text.split("\n")
        text_token = self.bert_embedding(text)
        text = np.array(text_token[0][1])
        if text.shape[0] > 14:
            text = text[:14, :]
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, "one_hot_mask must be nonnegative"
            mask = torch.zeros(
                (self.one_hot_mask, mask.shape[1], mask.shape[2])
            ).scatter_(0, mask.long(), 1)

        sample = {"label": mask, "text": text}

        return sample, mask_filename


# 新增，在Mixdataset中添加聚类ID的加载
class Mixdataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        row_text: str,
        joint_transform: Callable = None,
        one_hot_mask: int = False,
        cluster_id_path: str = None,
        load_text: bool = True,
    ) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, "img")
        self.output_path = os.path.join(dataset_path, "labelcol")
        self.images_list = natsort.natsorted(os.listdir(self.input_path))
        self.mask_list = natsort.natsorted(os.listdir(self.output_path))

        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        # 控制是否加载 HuggingFace 文本模型（在仅需图像特征时可以禁用以节省时间/内存）
        self.load_text = load_text
        if self.load_text:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT"
            )
            self.bert_embedding = AutoModel.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT"
            )
        else:
            self.tokenizer = None
            self.bert_embedding = None

        self.image_paths = [os.path.join(self.input_path, p) for p in self.images_list]
        if joint_transform:
            self.joint_transform = joint_transform
            self.img_size = joint_transform.output_size
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

        self.cluster_id_map = None  # 添加属性以存储聚类ID映射
        # 如果提供了cluster_id_path，则加载映射文件
        if cluster_id_path and os.path.exists(cluster_id_path):
            with open(cluster_id_path, "r") as f:
                self.cluster_id_map = json.load(f)  # 文件名到cluster_id的映射

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_name = self.images_list[idx]
        if "monu" in self.dataset_path:
            mask_name = img_name[:-3] + "png"
        else:
            mask_name = "mask_" + img_name  # qata_cov19
        image = cv2.imread(os.path.join(self.input_path, img_name))
        image = cv2.resize(image, self.img_size)
        mask = cv2.imread(os.path.join(self.output_path, mask_name), 0)
        mask = cv2.resize(mask, self.img_size)
        mask[mask <= 0] = 0
        mask[mask > 0] = 1
        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        # 安全读取文本，如果没有对应项则回退到空字符串
        text = self.rowtext.get(mask_name, "")
        text = text.split("\n") if isinstance(text, str) else []

        # 新增: 获取当前样本的cluster_id
        cluster_id = 0
        if self.cluster_id_map is not None:
            filename = os.path.splitext(img_name)[0]
            cluster_id = self.cluster_id_map.get(filename, 0)  # 默认cluster_id为0

        # 如果没有加载文本模型，返回占位空特征（在只使用图像的流程中会被忽略）
        if (
            self.load_text
            and self.tokenizer is not None
            and self.bert_embedding is not None
        ):
            with torch.no_grad():
                text_inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=10,
                )
                text_token = self.bert_embedding(**text_inputs)
            text = text_token["last_hidden_state"].cpu().numpy().squeeze()
        else:
            text = np.zeros((1, 1))

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, "one_hot_mask must be nonnegative"
            mask = torch.zeros(
                (self.one_hot_mask, mask.shape[1], mask.shape[2])
            ).scatter_(0, mask.long(), 1)

        # 新增cluster_id字段
        sample = {
            "image": image,
            "label": mask,
            "text": text,
            "name": os.path.splitext(img_name)[0],
            "cluster_id": cluster_id,
        }
        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample
