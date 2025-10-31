import torch
from timer import timers


transStream = torch.cuda.Stream()

dim=8192 * 2
a1 = torch.zeros(dim, dim, device='cuda')
b1 = torch.zeros(dim, dim, device='cuda')
a2 = torch.zeros(dim, dim, device='cuda')
b2 = torch.zeros(dim, dim, device='cuda')
c = torch.zeros(dim, dim, device='cpu', pin_memory=True)

timers('overlap').start()
timers('mem').start()  

with torch.cuda.stream(transStream):
    e1 = c.to('cuda', non_blocking=True)
    e2 = c.to('cuda', non_blocking=True)
timers('mem').stop()

timers("compute").start()
d = a1 * b1

timers("compute").stop()

torch.cuda.synchronize()
timers('overlap').stop()


timers('sequential').start()
g = a2 * b2
h = c.to('cuda')

timers('sequential').stop()

memTime = timers('mem').elapsed()
computeTime = timers('compute').elapsed()
overlapTime = timers('overlap').elapsed()
sequentialTime = timers('sequential').elapsed()

print(f'mem time: {memTime:.4f} seconds')
print(f'compute time: {computeTime:.4f} seconds')
print(f'overlap time: {overlapTime:.4f} seconds')
print(f'Sequential time: {sequentialTime:.4f} seconds')


