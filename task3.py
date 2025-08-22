import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def hilbert_points(order: int, device=device):
    n = 1 << order                     
    m = n * n                         
    d = torch.arange(m, device=device, dtype=torch.int64)

    x = torch.zeros_like(d)
    y = torch.zeros_like(d)
    t = d.clone()

    s = 1
    while s < n:
        rx = (t // 2) & 1               
        ry = (t ^ rx) & 1               

        mask = (ry == 0)
        invmask = mask & (rx == 1)
        x = torch.where(invmask, s - 1 - x, x)
        y = torch.where(invmask, s - 1 - y, y)
        tmpx = x
        x = torch.where(mask, y, x)
        y = torch.where(mask, tmpx, y)

        x = x + s * rx
        y = y + s * ry

        t = t // 4
        s <<= 1

    pts = torch.stack([x, y], dim=1).to(torch.float32)
    if n > 1:
        pts = pts / (n - 1)
    return pts

order = 5
pts = hilbert_points(order).to(device)
xy = pts.detach().cpu().numpy()
plt.figure(figsize=(16, 10))
plt.plot(xy[:, 0], xy[:, 1], linewidth=0.8)
plt.show()
