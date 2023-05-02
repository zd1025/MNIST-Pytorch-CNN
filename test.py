import time
import torch
import torchvision
from torch import nn
from model import load_model
from tensorboardX import SummaryWriter
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# 模型存放路径
model_save_path = r'./save_model/MNIST_model_zd.pth'

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 如果有 gpu 那么以 gpu 进行训练 如果没有就使用 cpu 训练

# 文件路径
log_path = r'./logs' # 日志文件路径
dataset_path = r'./data' # 数据集存放路径

# 参数初始化
batch_size = 64 # 批处理的大小
worker_num = 1 # 同时运行进程数

# 可视化工具初始化
writer = SummaryWriter(log_path)

# 损失函数（交叉熵）
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 测试集加载
test_data = MNIST(dataset_path,train=False, transform=torchvision.transforms.ToTensor(),
                                            download=True)

# 数据集大小
test_data_size = len(test_data)

test_data_loader = DataLoader(test_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=worker_num)

if __name__ == '__main__':
    stime = time.time()

    # 加载模型
    model = load_model(model_save_path)

    # 测试步骤开始
    model.eval()
    print("测试数据集的长度为：{}".format(test_data_size))
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        step = 0
        for data in test_data_loader:
            imgs, targets = data
            imgs.to(device)
            targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
            writer.add_images("测试图片", imgs, step)
            step = step + 1

    print("测试集上的Loss总和：{}".format(total_test_loss))
    print("测试集上的正确率：{}".format(total_accuracy / test_data_size * 100))
    writer.add_scalar("test_loss", total_test_loss, 0)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, 0)
    etime = time.time()
    print("模型测试花费了{}秒".format(etime - stime))
    writer.close()