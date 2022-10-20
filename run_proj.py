import os
import numpy as np

for w_avg in [1, 10, 1000, 10000]:
    for i in range(10):
        os.system(f"python projector.py --outdir=out_proj/loss_proj_s/ --w-avg-samples {w_avg} --seed {i}" +
        " --target=ffhq.png --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl --save-video False --proj-space s")


w_avg = [1, 10, 1000, 10000]
loss = np.empty((len(w_avg), 10))
for w_avg_idx in range(len(w_avg)):
    for seed_idx in range(10):
        loss[w_avg_idx][seed_idx] = np.load(f'out_proj/loss_proj_s/loss_{w_avg[w_avg_idx]}_samples_{seed_idx}.npy')

loss_std = np.std(loss, axis=1)
print(np.min(loss, axis=1))