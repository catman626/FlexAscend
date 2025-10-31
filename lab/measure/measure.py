import torch
from timer import timers

dim = 512
bs = 32

#warmup
def cudaWarmup():
    a = torch.randn(bs, dim, dim, device='cuda')
    b = torch.zeros(bs, dim, dim, device='cuda')
    c = torch.bmm(a, b)


def testCompute(bs):
    a = torch.randn(bs, dim, dim, device='cuda')
    b = torch.zeros(bs, dim, dim, device='cuda')
    
    timers(str(bs)).start()
    c = torch.bmm(a, b)
    a = torch.bmm(b, c)
    b = torch.bmm(c, a)

    timers(str(bs)).stop()

def testCommunicate(bs):
    a = torch.randn(bs, dim, dim, device='cpu')

    timers(str(bs)).start()
    b = a.to('cuda')
    timers(str(bs)).stop()

def measureTime(mode):
    stride = 64 
    steps = 32
    for i in range(1, steps + 1):
        bs = i * stride

        if mode == "compute":
            testCompute(bs)
        elif mode == "communicate":
            testCommunicate(bs)

    for i in range(1, steps + 1):
        bs = i * stride
        t = timers(str(bs)).elapsed()
        print(f"batch size: {bs}, time: {t:.4f} seconds")
    

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, 
                        default="compute", 
                        choices=["compute", "communicate"],)

    args = parser.parse_args()

    cudaWarmup()

    measureTime(args.mode)
