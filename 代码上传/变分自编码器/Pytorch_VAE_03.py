#注意更换输出重建；输入数据集的绝对路径
#输入数据集路径：root_dir
#代码调用了CUDA进行并行运算

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

# 检查CUDA是否可用，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 数据集类
class BeeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('L')  # 转换为灰度图
        if self.transform:
            image = self.transform(image)
        return image

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# 加载数据集
dataset = BeeDataset(root_dir='I:/dataset/SynologyDrive/肝脏超声8大器官分割/aus2rus/aus2rus/trainA', transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 定义损失函数
def loss_fn(recon_x, x, mean, log_var):
    recon_x = recon_x.view(-1, 28 * 28)
    x = x.view(-1, 28 * 28)
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (BCE + KLD) / x.size(0)

# 定义编码器
class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_size):
        super(Encoder, self).__init__()
        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x):
        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

# 定义解码器
class Decoder(nn.Module):
    def __init__(self, layer_sizes, latent_size):
        super(Decoder, self).__init__()
        self.MLP = nn.Sequential()
        input_size = latent_size
        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z):
        x = self.MLP(z)
        return x

class VAE(nn.Module):
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(encoder_layer_sizes, latent_size)
        self.decoder = Decoder(decoder_layer_sizes, latent_size)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(-1, 28 * 28)
        means, log_var = self.encoder(x)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z)
        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss_fn(self, recon_x, x, mean, log_var):
        return loss_fn(recon_x, x, mean, log_var)

    def inference(self, z):
        recon_x = self.decoder(z)
        return recon_x

# 定义模型的层大小和潜在空间的维度
encoder_layer_sizes = [784, 400, 200]  # 编码器层大小
latent_size = 20  # 潜在空间的维度
decoder_layer_sizes = [200, 400, 784]  # 解码器层大小

# 初始化模型并移动到GPU
model = VAE(encoder_layer_sizes, latent_size, decoder_layer_sizes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
num_epochs = 600  # 根据需要调整迭代次数
loss_values = []  # 用于记录每个batch的损失值
for epoch in range(num_epochs):
    for batch_idx, data in enumerate(dataloader):
        # 将数据移动到GPU
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var, z = model(data)
        loss = model.loss_fn(recon_batch, data, mu, log_var)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())  # 记录损失值
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

# 绘制损失值图表
plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Training Loss')
plt.title('Training Loss Over Batches')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 生成结果
with torch.no_grad():
    z = torch.randn(64, latent_size).to(device)  # 随机采样隐空间的点并移动到GPU
    sample = model.inference(z)
    sample = sample.view(-1, 1, 28, 28)  # 调整形状以便于可视化

# 显示生成的图片
def show_image(tensor):
    tensor = tensor.cpu().clone()  # 将张量移回CPU
    tensor = tensor * 0.5 + 0.5  # 反归一化
    tensor = tensor.squeeze().numpy()  # 转换为numpy数组并去除单维度
    plt.imshow(tensor, cmap='gray')  # 使用灰度色图显示图像
    plt.axis('off')  # 不显示坐标轴
    plt.show()

show_image(sample[0])

print("训练完成，生成的图片已显示。")
