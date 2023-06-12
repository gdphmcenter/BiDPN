import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_size, noise=False):
        super(Encoder, self).__init__()
        self.latent_size = latent_size

        if noise:
            self.latent_size *= 2

        # 初始x = [270,1,4500,1]
        self.main1 = nn.Sequential(
            # layer1 x = [270,1,4500,1]
            nn.Conv2d(1, 2, (4, 1), stride=(2, 1)),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(),

            # layer2 x = [270,2,2249,1]
            nn.Conv2d(2, 4, (4, 1), stride=(2, 1)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),

            # layer3 x = [270, 4, 1123, 1]
            nn.Conv2d(4, 8, (3, 1), stride=(2, 1)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),

            # layer4 x = [270, 8, 561, 1]
            nn.Conv2d(8, 16, (4, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            # layer5 x = [270, 8, 561, 1]
            nn.Conv2d(16, 32, (3, 1), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            # layer6 x = [270, 32, 139, 1]
            nn.Conv2d(32, 64, (4, 1), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            # layer7 x = [270, 64, 68, 1]
            nn.Conv2d(64, 128, (3, 1), stride=(2, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            # layer8 x = [270, 128, 33, 1])
            nn.Conv2d(128, 256, (4, 1), stride=(2, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            # layer9 x = [270, 256, 15, 1]
            nn.Conv2d(256, 512, (3, 1), stride=(2, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            # layer10 x = [270, 512, 7, 1]
            nn.Conv2d(512, 1024, (4, 1), stride=(2, 1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),

            # layer11 x = [270, 1024, 2, 1]
            nn.Conv2d(1024, 2048, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(),

            # layer12 x = [270, 2048, 1, 1]
            nn.Conv2d(2048, self.latent_size, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(self.latent_size),
            nn.LeakyReLU(),
            # layer12 x = [270, 100, 1, 1]
        )

    def forward(self, input):
        input = input.unsqueeze(1).unsqueeze(3)
        output = self.main1(input)
        return output



class Generator(nn.Module):
    def __init__(self, latent_size, noise=False):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.main = nn.Sequential(
            # layer1 x = [270, 100, 1, 1]
            nn.ConvTranspose2d(self.latent_size, 512, (3, 1), stride=(2, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            # layer2 x = [270, 512, 3, 1]
            nn.ConvTranspose2d(512, 256, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            # layer3 x = [270, 512, 6, 1]
            nn.ConvTranspose2d(256, 128, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            # layer4 x = [270, 128, 12, 1]
            nn.ConvTranspose2d(128, 64, (3, 1), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            # layer5 x = [270, 64, 25, 1]
            nn.ConvTranspose2d(64, 32, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            # layer6 x = [270, 32, 50, 1]
            nn.ConvTranspose2d(32, 16, (3, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            # layer7 x = [270, 8, 101, 1]
            nn.ConvTranspose2d(16, 8, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),

            # layer8 x = [270, 8, 202, 1]
            nn.ConvTranspose2d(8, 4, (3, 1), stride=(2, 1)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),

            # layer9 x = [270, 4, 405, 1]
            nn.ConvTranspose2d(4, 2, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(),

            # layer10 x = [270, 2, 810, 1]
            nn.ConvTranspose2d(2, 1, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),

            # layer10 x = [270, 1, 1620, 1]
        )
        self.linear = nn.Linear(1620, latent_size)

    def forward(self, input):
        input = torch.reshape(input, (-1, self.latent_size, 1, 1))
        output = self.main(input)

        output = output.squeeze()
        output = self.linear(output)
        return output

class Discriminator(nn.Module):
    def __init__(self, latent_size, dropout, output_size=10):
        super(Discriminator, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size

        self.infer_x = nn.Sequential(
            nn.Conv2d(self.latent_size, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(512, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_z = nn.Sequential(
            nn.Conv2d(self.latent_size, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(512, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_joint = nn.Sequential(
            nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
            nn.LeakyReLU(),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
            nn.LeakyReLU(),
            nn.Dropout2d(p=self.dropout)
        )

        self.final = nn.Conv2d(1024, self.output_size, 1, stride=1, bias=True)

    def forward(self, x, z):
        output_x = self.infer_x(x)
        output_z = self.infer_z(z)
        output_features = self.infer_joint(torch.cat([output_x, output_z], dim=1))
        output = self.final(output_features)
        if self.output_size == 1:
            output = F.sigmoid(output)
        return output.squeeze()


if __name__ == "__main__":
    data = torch.rand(270, 4500)
    #data = torch.reshape(data, (270, 1, 4500, 1))
    print(data.shape)

    latentsize = 4500
    batch_size = 270
    en = Encoder(latentsize, False)
    output = en(data)
    print(output.shape)

    data = torch.rand(270, 4500)
    g = Generator(latentsize)
    output_g = g(data)
    print(output_g.shape)

    # ----------------------discriminate----------------------------------
    output_g = output_g.view(batch_size, latentsize, 1, 1)
    output = output.view(batch_size, latentsize, 1, 1)
    D = Discriminator(latentsize, 0.2, 1)
    p = D(output, output_g)
    print(p)
