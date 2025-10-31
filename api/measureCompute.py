import torch
from timer import timers

dim = 512
bs = 32


#warmup
a = torch.randn(32, dim, dim, device='cuda')
b = torch.zeros(32, dim, dim, device='cuda')
c = torch.bmm(a, b)

stride = 32
for i in range(1, 32 + 1):
    bs = i * stride
    a = torch.randn(bs, dim, dim, device='cuda')
    b = torch.zeros(bs, dim, dim, device='cuda')
    
    timers(f"compute-{bs}").start()
    c = torch.bmm(a, b)
    timers(f'compute-{bs}').stop()


timeRecord = dict()
for i in range(1, 32 + 1):
    bs = i * stride
    t = timers(f"compute-{bs}").elapsed()

    timeRecord[bs] = t


for bs, t in timeRecord.items():
    print(f"batch size: {bs}, time: {t:.4f} seconds")

    