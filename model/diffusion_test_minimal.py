import torch
import torch.nn as nn
import torch.optim as optim

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Contracting Path
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Expansive Path
        self.up3 = self.up_conv(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = self.up_conv(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = self.up_conv(128, 64)
        self.dec1 = self.conv_block(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Contracting Path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Expansive Path
        dec3 = self.up3(enc4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.out(dec1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

def add_noise(x, noise_scale):
    return x + noise_scale * torch.randn_like(x)

def diffusion_step(model, x_noisy, beta):
    prediction = model(x_noisy)
    x_noisy = (1 - beta) * x_noisy + beta * prediction
    return x_noisy

def train(model, data, epochs=100, lr=0.0001, beta=0.1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        for x in data:
            x_noisy = add_noise(x, noise_scale=0.1)
            optimizer.zero_grad()
            x_pred = diffusion_step(model, x_noisy, beta)
            loss = loss_fn(x_pred, x)
            print('loss: ', loss)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Dummy Data (you can replace this with actual data)
data = [torch.randn(1, 3, 32, 32) for _ in range(100)]

# Create and train the model
model = UNet(in_channels=3, out_channels=3)
train(model, data)
