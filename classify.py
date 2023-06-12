import argparse
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
from model1 import *
import os
from pathlib import Path
from clusterone import get_data_path, get_logs_path
from dataset import Mydataset

CLUSTERONE_USERNAME = "gaurav9310"

batch_size = 50
lr = 1e-3
latent_size = 4500
num_epochs = 7500
cuda_device = "0"


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default="./datasets", help='path to dataset')
parser.add_argument('--use_cuda', type=boolean_string, default=True)
parser.add_argument('--save_model_dir', default='./model')
parser.add_argument('--save_image_dir', default='./image')
parser.add_argument('--reuse', type=boolean_string, default=False)
parser.add_argument('--save_freq', type=int, default=1)

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
print(opt)


def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


# 数据集
train_loader = torch.utils.data.DataLoader(
    Mydataset(150),  # 150代表选取多少的正常数据
    batch_size=batch_size, shuffle=True)
# 保存模型的地方
save_image_dir = get_logs_path(opt.save_image_dir)
save_model_dir = get_logs_path(opt.save_model_dir)
# 创建模型
netE = tocuda(Encoder(latent_size, False))
netG = tocuda(Generator(latent_size))
netD = tocuda(Discriminator(latent_size, 0.2, 1))

# 加载参数
if opt.reuse:
    for epoch in range(num_epochs):
        if epoch % opt.save_freq == 0:
            model_file = Path("%s/netG_epoch_%d.pth" % (save_model_dir, epoch))
            if model_file.is_file():
                continue
            else:
                break
    epoch = epoch - 1 * opt.save_freq
    if epoch == -1 * opt.save_freq:
        netE.apply(weights_init)
        netG.apply(weights_init)
        netD.apply(weights_init)
        print("No saved models found to resume from. Starting from scratch.")
    else:
        print("Loading models saved after epochs : ", epoch + 1)
        encoder_state_dict = torch.load("%s/netE_epoch_%d.pth" % (save_model_dir, epoch))
        generator_state_dict = torch.load("%s/netG_epoch_%d.pth" % (save_model_dir, epoch))
        discriminator_state_dict = torch.load("%s/netD_epoch_%d.pth" % (save_model_dir, epoch))

        netE.load_state_dict(encoder_state_dict)
        netG.load_state_dict(generator_state_dict)
        netD.load_state_dict(discriminator_state_dict)
else:
    # 初始化模型
    netE.apply(weights_init)
    netG.apply(weights_init)
    netD.apply(weights_init)

# 建立优化器和损失函数
optimizerG = optim.Adam([{'params': netE.parameters()},
                         {'params': netG.parameters()}], lr=lr)
optimizerD = optim.Adam(netD.parameters(), lr=lr)
gamma = 0.96
g_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=gamma)
d_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=gamma)
criterion = tocuda(nn.BCELoss())

for epoch in range(num_epochs):
    i = 0
    for (data, target) in train_loader:
        # label用于辨别器的loss计算
        real_label = Variable(tocuda(torch.ones(batch_size)))
        fake_label = Variable(tocuda(torch.zeros(batch_size)))

        # d_real = [90,4500]
        d_real = Variable(tocuda(data))  # real
        # z_fake = [90,4500,1,1]
        z_fake = Variable(tocuda(torch.randn(batch_size, latent_size, 1, 1)))
        # d_fake = [90,4500]
        d_fake = netG(z_fake)

        # z_real = [90,9000]
        z_real = netE(d_real)
        z_real = z_real.view(batch_size, -1)

        output_real = netD(d_real.unsqueeze(2).unsqueeze(3), z_real.view(batch_size, latent_size, 1, 1))
        output_fake = netD(d_fake.unsqueeze(2).unsqueeze(3), z_fake)

        loss_g = criterion(output_fake, real_label) + criterion(output_real, fake_label)
        loss_d = criterion(output_fake, fake_label) + criterion(output_real, real_label)

        optimizerG.zero_grad()
        optimizerD.zero_grad()

        loss_g.backward(retain_graph=True)
        loss_d.backward()

        optimizerD.step()
        optimizerG.step()

        if i % 100 == 0:
            print("Epoch :", epoch, "Iter :", i, "D Loss :", loss_d.item(), "G loss :", loss_g.item(),
                  "D(x) :", output_real.mean().item(), "D(G(x)) :", output_fake.mean().item())
            g_scheduler.step()
            d_scheduler.step()
        i += 1

    if epoch % opt.save_freq == 0:
        torch.save(netG.state_dict(), "%s/netG_epoch_%d.pth" % (save_model_dir, epoch))
        torch.save(netE.state_dict(), "%s/netE_epoch_%d.pth" % (save_model_dir, epoch))
        torch.save(netD.state_dict(), "%s/netD_epoch_%d.pth" % (save_model_dir, epoch))
