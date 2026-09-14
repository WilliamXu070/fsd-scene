import argparse
from fsd.runtime import load_config
from fsd.engine import train

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/baseline.yaml')
    parser.add_argument('--stage',choices=['a','b'],default='a')
    parser.add_argument('--run',required=True)
    starting = parser.add_mutually_exclusive_group()
    starting.add_argument('--resume')
    starting.add_argument('--initialize', help='Fresh recorded experiment: load weights only, reset all training state')
    parser.add_argument('--max-updates',type=int)
    parser.add_argument('--indices',help='Comma-separated real training indices for diagnostic overfit only')
    args=parser.parse_args()
    indices=[int(x) for x in args.indices.split(',')] if args.indices else None
    train(load_config(args.config),args.run,args.stage,args.resume,args.max_updates,indices,initialize=args.initialize)
