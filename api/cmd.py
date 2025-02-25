import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", nargs="+", help="provode a list of ckpt files")
     
    args = parser.parse_args()
    for ckpt in args.ckpt:
        print(ckpt)