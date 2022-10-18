import copy
import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
from training import networks

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--edition-layer', 'edition_layer', help='layer of style channel to edit', type=int, default=24)
@click.option('--edition-channel', 'edition_channel', help='style channel to edit', type=int, default=31)
@click.option('--alpha', help='quantiity to add or retrieve', type=float, default=10)
@click.option('--projected-s', help='Projection result file from S', type=str, metavar='FILE')
@click.option('--projected-w', help='Projection result file from W or W+', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_edited_images(
    ctx: click.Context,
    network_pkl: str,
    noise_mode: str,
    edition_layer: int,
    edition_channel: int,
    alpha: float,
    projected_s: Optional[str],
    projected_w: Optional[str],
    outdir: str
):
    
    # Array with nb of channels per layer of the synthetizer
    layers = [512 for _ in range(15)]
    layers.extend([256, 256, 256, 128, 128, 128, 64, 64, 64, 32, 32])
    layers = np.array(layers)
    
    assert edition_layer in [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24], 'layer must be a conv, not RGB'
    assert edition_channel < layers[edition_layer], f'there are {layers[edition_layer]} channels in layer {edition_layer}'
    
    os.makedirs(outdir, exist_ok=True)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G_w = legacy.load_network_pkl(f)['G_ema'].to(device)

    # Load G that will generate image from style vector
    G = networks.Generator_from_S(G_w.z_dim, G_w.c_dim, G_w.w_dim, G_w.img_resolution, G_w.img_channels).eval().requires_grad_(False).to(device)
    G.load_state_dict(G_w.state_dict())
    
    if projected_w is not None:
        # Load G that will generate style vector from w
        G_out_S = networks.Generator_out_S(G_w.z_dim, G_w.c_dim, G_w.w_dim, G_w.img_resolution, G_w.img_channels).eval().requires_grad_(False).to(device)
        G_out_S.load_state_dict(G_w.state_dict())

        # Get s from projected_w
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        s_list = G_out_S.synthesis(ws, noise_mode=noise_mode)

        # Concatenate style vectors to make one vector
        s = np.concatenate((s_list[0].detach().cpu().numpy(), s_list[1].detach().cpu().numpy()), axis=1)
        for idx in range (2, len(s_list)):
            s = np.concatenate((s, s_list[idx].detach().cpu().numpy()), axis=1)

    else:
        s = np.load(projected_s)['s']

    # Edit style vector
    if edition_layer == 0:
        edition_idx = edition_channel
    else:
        edition_idx = np.sum(layers[:edition_layer]) + edition_channel

    old_val = s[0, edition_idx]
    s[0, edition_idx] *= alpha
    new_val = s[0, edition_idx]
    print(f'Channel {edition_channel} of layer {edition_layer} (i.e. channel {edition_idx}) went from {old_val} to {new_val}')

    s_edited_tensor = torch.tensor(s, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    for idx, s_ in enumerate(s_edited_tensor):
        img = G.synthesis(s_.unsqueeze(0), noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        
#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_edited_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------