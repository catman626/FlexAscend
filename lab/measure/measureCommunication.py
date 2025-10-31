import torch

from timer import timers

dim=512
a = torch.zeros(dim, dim, device='cpu')

steps = 32
stride = 32
for i in range(1, steps):
    
    bs = stride * i
    timers(f"bs-{bs}").start()
    b = a.to('cuda')
    timers(f"bs-{bs}").stop()


for i in range(1, steps):
    bs = stride * i
    t = timers(f"bs-{bs}").elapsed()
    print(f"batch size: {bs}, time: {t} seconds")
