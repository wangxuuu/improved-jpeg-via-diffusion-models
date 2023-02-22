# main function for running JSCC model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from itertools import chain
from models.jscc import *
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import os
import argparse
import functools
from models.samplers import *

# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_model(encoder, decoder):
    print("============== Encoder ==============")
    print(encoder)
    print("============== Decoder ==============")
    print(decoder)
    print("")


def create_model(in_channels=3,
                c=64,
                kernel_size=5,
                dilation=(1,1),
                channel_type='awgn',
                snr=0.0):
    autoencoder = Autoencoder(in_channels=in_channels,
                c=c,
                kernel_size=kernel_size,
                dilation=dilation,
                channel_type=channel_type,
                snr=snr)
    print_model(autoencoder.encoder, autoencoder.decoder)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder


def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

    Returns:
    The standard deviation.
    """
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

sigma =  25.0#@param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)

def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument("--valid", action="store_true", default=False,
                        help="Perform validation only.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--c_out", type=int, default=8, help="the channel of the latent representation; used to control the compression ratio:  k/n=c_out/(16*3)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train for.")
    parser.add_argument("--sample_dir", type=str, default='./results/jscc_samples/')
    parser.add_argument("--dataset_dir", type=str, default='../data/')
    parser.add_argument("--ckpt_dir", type=str, default='./checkpoints/')
    parser.add_argument("--snr", type=float, default=0, help="SNR of the channel")
    parser.add_argument("--snr_type", type=str, default='awgn', help="Type of the channel: awgn or fading")
    parser.add_argument("--forward_steps", type=int, default=100, help="Number of forward steps in the diffusion model")
    parser.add_argument("--select_t", type=float, default=0.3, help="Select the t-th step in the diffusion model")
    args = parser.parse_args()
    if args.snr > 0:
        args.ckpt_dir = args.ckpt_dir + args.snr_type + str(args.snr) + '/'
    # Create model
    autoencoder = create_model(in_channels=1, c=args.c_out, channel_type=args.snr_type, snr=args.snr)

    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    trainset = torchvision.datasets.MNIST(root=args.dataset_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root=args.dataset_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if args.valid:
        print("Loading checkpoint...")
        autoencoder.load_state_dict(torch.load(args.ckpt_dir))

    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    # if not os.path.exists(args.sample_dir):
    #     os.mkdir(args.sample_dir)

    # Define an optimizer and criterion
    # criterion = nn.BCELoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters())

    for epoch in range(args.epochs):
        running_loss = 0.0

        for i, (inputs, y) in enumerate(trainloader, 0):
            inputs = get_torch_vars(inputs)
            y = get_torch_vars(y)
            diffused_inputs = forward_process(inputs, marginal_prob_std_fn, args.forward_steps, device=device, end_t=args.select_t, only_final=True)
            # ============ Forward ============
            # encoded = autoencoder.encode(inputs)
            # outputs = autoencoder.decode(encoded)
            outputs = autoencoder(diffused_inputs)
            loss = criterion(outputs, diffused_inputs)

            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # ============ Logging ============
            running_loss += loss.data
            if (i+1) % 200 == 0:
                print('Epoch [%d, %5d] Step [%d, %5d] loss: %.3f ' %
                      (epoch + 1, args.epochs, i+1, len(trainloader), running_loss / 200))
                running_loss = 0.0

        # if (epoch+1) % 10 == 0:
        #     with torch.no_grad():
        #         # Save the reconstructed images
        #         x_concat = torch.cat([inputs, outputs], dim=3)
        #         save_image(x_concat, os.path.join(args.sample_dir, 'reconst-{}.png'.format(epoch+1)), nrow=4)


    print('Finished Training')
    print('Saving Model...')

    torch.save(autoencoder.state_dict(), os.path.join(args.ckpt_dir+'jscc.pt'))


if __name__ == '__main__':
    main()