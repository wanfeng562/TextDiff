{
  "exp_dir":  "saved_textdiff/monuseg_2",   # 实验保存目录
  "model_type": "ddpm",  # 模型类型
  "category": "monuseg_2",  # 数据集类别
  "number_class": 2,  # 类别数
  "ignore_label": 255, # 需要忽略的像素标签值

  "training_path": "datasets/monuseg_2/Train_Folder", # 训练集路径
  "validation_path": "datasets/monuseg_2/Val_Folder", # 验证集路径
  "testing_path": "datasets/monuseg_2/Test_Folder",   # 测试集路径
  "model_path": "checkpoints/ddpm/256x256_diffusion_uncond.pt", # 预训练模型路径

  "image_size": 256,  # 图像尺寸
  "dim": [512, 512, 256, 256],  # 网络维度
  "steps": [50, 150, 250], # 扩散步骤
  "blocks": [6, 8, 12, 16], # 残差块数量

  "model_num": 1, # ?
  "batch_size": 1,  # 批次大小
  "max_training": 50, # 最大训练轮次

  "training_number": 50,  # 训练次数
  "testing_number": 20,   # 测试次数

  "upsample_mode":"bilinear", # 上采样方式
  "share_noise": true,  # ?
  "input_activations": false  # ?
 }

