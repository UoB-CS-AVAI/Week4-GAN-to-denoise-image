import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

from utils.GAN.generator import Generator
from utils.GAN.discriminator import Discriminator
from utils.GAN.loss import discriminator_loss, generator_loss
from utils.GAN.weights_init import weights_init

### Challenge - Task3: MSE and PSNR computation for the denoised image and the original image
### Challenge - Task4: MSE and PSNR computation for the noisy image and the original image
def compute_mse(img1, img2):
    return F.mse_loss(img1, img2)
def compute_psnr(mse):
    mse_tensor = torch.tensor(mse)  # Convert the float to a tensor
    return 10 * torch.log10(1 / mse_tensor)

# Step 4 - training loop
def training_loop(clean_imgs):
    clean_imgs = clean_imgs.to(device)
    noise = 0.3 * torch.randn(clean_imgs.shape).to(device)  # generate noise
    noisy_imgs = clean_imgs + noise
    noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)

    '''Training step for the Discriminator'''
    generated_imgs = generator(noisy_imgs.view(noisy_imgs.size(0), -1)).view(noisy_imgs.size(0), 1, 28, 28)
    real_output = discriminator(clean_imgs)
    fake_output = discriminator(generated_imgs.detach())
    loss_D = discriminator_loss(real_output, fake_output, device)

    optim_D.zero_grad()
    loss_D.backward()
    optim_D.step()

    '''Training step for the Generator'''
    generated_imgs = generator(noisy_imgs.view(noisy_imgs.size(0), -1)).view(noisy_imgs.size(0), 1, 28, 28)
    fake_output = discriminator(generated_imgs)
    loss_G = generator_loss(fake_output, clean_imgs, generated_imgs )

    optim_G.zero_grad()
    loss_G.backward()
    optim_G.step()

    return loss_D.item(), loss_G.item(), noisy_imgs, clean_imgs


# Main function
if __name__ == "__main__":
    # use gpu or cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 1 - load MNIST dataset load MNIST dataset
    train_dataset = datasets.MNIST(root="./data/", train=True, transform=transforms.ToTensor(), download=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    image_size = 28 * 28  # MNIST image size

    # Step 2 - instantiate the generator and discriminator
    generator = Generator(input_dim=image_size).to(device)
    discriminator = Discriminator().to(device)

    # Step 3 - define separate Adam optimizers
    lr = 0.0001
    optim_G = optim.Adam(generator.parameters(), lr=lr)
    optim_D = optim.Adam(discriminator.parameters(), lr=lr)

    # learning rate scheduler
    scheduler_G = torch.optim.lr_scheduler.StepLR(optim_G, step_size=5, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optim_D, step_size=5, gamma=0.5)

    # save Directory
    checkpoint_dir = './training_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    image_dir = './denoised_images'
    os.makedirs(image_dir, exist_ok=True)

    # Initialisation of the network weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Step 4 - training loop
    # def training_loop(clean_imgs)

    # Step 5 - train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss_D, total_loss_G = 0, 0
        for batch_idx, (clean_imgs, _) in enumerate(train_loader):
            # Get noisy_imgs and clean_imgs
            loss_D, loss_G, noisy_imgs, clean_imgs = training_loop(clean_imgs)
            total_loss_D += loss_D
            total_loss_G += loss_G

            # Save only the first batch of images for visualisation
            if batch_idx == 0:
                # Save noisy images
                noisy_img_np = noisy_imgs[0].cpu().numpy()
                plt.imshow(noisy_img_np[0], cmap='gray')  # (channels, height, width) from torch, only 1 channel, because the image is greyscale
                plt.title('Noisy Image')
                plt.savefig(f'{image_dir}/epoch_{epoch + 1}_noisy.png')
                plt.close()

                # Save denoised images
                denoised_img = generator(noisy_imgs.view(noisy_imgs.size(0), -1)).view(noisy_imgs.size(0), 1, 28, 28)
                denoised_img = torch.clamp(denoised_img, 0., 1.)
                denoised_img_np = denoised_img[0].detach().cpu().numpy()
                plt.imshow(denoised_img_np[0], cmap='gray')
                plt.title('Denoised Image')
                plt.savefig(f'{image_dir}/epoch_{epoch + 1}_denoised.png')
                plt.close()

                # Save clean images
                clean_img_np = clean_imgs[0].cpu().numpy()
                plt.imshow(clean_img_np[0], cmap='gray')
                plt.title('Clean Image')
                plt.savefig(f'{image_dir}/epoch_{epoch + 1}_clean.png')
                plt.close()

                # Challenge - Compute MSE and PSNR for the noisy and denoised images
                mse_noisy = compute_mse(noisy_imgs, clean_imgs).item()
                mse_denoised = compute_mse(denoised_img, clean_imgs).item()
                psnr_noisy = compute_psnr(mse_noisy)
                psnr_denoised = compute_psnr(mse_denoised)

                # Print MSE and PSNR values
                print(f'Epoch {epoch + 1}/{num_epochs}, MSE and PSNR for noisy and original images: MSE = {mse_noisy}, PSNR = {psnr_noisy}')
                print(f'Epoch {epoch + 1}/{num_epochs}, MSE and PSNR for the denoised and original images: MSE = {mse_denoised}, PSNR = {psnr_denoised}')

        # Learning rate updated
        scheduler_G.step()
        scheduler_D.step()

        # Print the average loss per epoch
        print(f'After Epoch [{epoch + 1}/{num_epochs}], D_loss: {total_loss_D / len(train_loader):.4f}, G_loss: {total_loss_G / len(train_loader):.4f}')

    print("Training completed!")
