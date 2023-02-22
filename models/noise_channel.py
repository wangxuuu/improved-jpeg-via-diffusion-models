import torch
import numpy as np

"""
implement the noise channel in JSCC
Ref : https://github.com/kurka/deepJSCC-feedback/
"""

def real_awgn(x, stddev):
    """Implements the real additive white gaussian noise channel.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # additive white gaussian noise
    awgn = torch.randn(x.shape).to(x.device) * stddev
    y = x + awgn

    return y


def fading(x, stddev, h=None):
    """Implements the fading channel with multiplicative fading and
    additive white gaussian noise.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # channel gain
    if h is None:
        h = torch.complex(
            torch.randn([x.shape[0], 1]).to(x.device) * 1 / np.sqrt(2),
            torch.randn([x.shape[0], 1]).to(x.device) * 1 / np.sqrt(2),
        )
    # additive white gaussian noise
    awgn = torch.complex(
        torch.randn(x.shape).to(x.device) * 1 / np.sqrt(2),
        torch.randn(x.shape).to(x.device) * 1 / np.sqrt(2),
    )

    return (h * x + stddev * awgn), h


def phase_invariant_fading(x, stddev, h=None):
    """Implements the fading channel with multiplicative fading and
    additive white gaussian noise. Also assumes that phase shift
    introduced by the fading channel is known at the receiver, making
    the model equivalent to a real slow fading channel.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # channel gain
    if h is None:
        n1 = torch.randn((x.shape[0],1)).to(x.device) * 1 / np.sqrt(2)
        n2 = torch.randn((x.shape[0],1)).to(x.device) * 1 / np.sqrt(2)
        h = torch.sqrt(torch.square(n1) + torch.square(n2))

    # additive white gaussian noise
    awgn = torch.randn(x.shape).to(x.device) * stddev / np.sqrt(2)

    return (h * x + awgn), h


class Channel():
    def __init__(self, channel_type, channel_snr):
        super(Channel, self).__init__()
        self.channel_type = channel_type
        self.channel_snr = channel_snr

    def __call__(self, inputs):
        (encoded_img, prev_h) = inputs
        inter_shape = encoded_img.shape
        # reshape array to [-1, dim_z]
        z = encoded_img.flatten(1)
        # convert from snr to std
        # print("channel_snr: {}".format(self.channel_snr))
        noise_stddev = np.sqrt(10 ** (-self.channel_snr / 10))

        # Add channel noise
        if self.channel_type == "awgn":
            dim_z = z.shape[1]
            # normalize latent vector so that the average power is 1
            z_in = np.sqrt(dim_z) * torch.nn.functional.normalize(z, dim=1)
            z_out = real_awgn(z_in, noise_stddev)
            h = torch.ones_like(z_in)  # h just makes sense on fading channels

        elif self.channel_type == "fading":
            dim_z = z.shape[1]//2
            # convert z to complex representation
            z_in = torch.complex(z[:, :dim_z], z[:, dim_z:])
            # normalize the latent vector so that the average power is 1
            z_norm = torch.sum(torch.real(z_in * torch.conj(z_in)), dim=1, keepdim=True)
            z_in = z_in * torch.complex(torch.sqrt(torch.tensor(dim_z).to(z.device) / z_norm), torch.tensor(0.0).to(z.device))

            z_out, h = fading(z_in, noise_stddev, prev_h)
            # convert back to real
            z_out = torch.cat([torch.real(z_out), torch.imag(z_out)], 1)

        elif self.channel_type == "fading-real":
            # half of the channels are I component and half Q
            dim_z = z.shape[1] // 2
            # normalization
            z_in = np.sqrt(dim_z) * torch.nn.functional.normalize(z, dim=1)
            z_out, h = phase_invariant_fading(z_in, noise_stddev, prev_h)

        else:
            raise Exception("This option shouldn't be an option!")

        # convert signal back to intermediate shape
        z_out = z_out.reshape(inter_shape)
        # compute average power
        avg_power = torch.mean(torch.real(z_in * torch.conj(z_in)))
        # add avg_power as layer's metric
        return z_out, avg_power, h