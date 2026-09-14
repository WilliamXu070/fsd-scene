import argparse
import torch
from fsd.sealing import freeze_selection

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--validation-report',required=True)
    p.add_argument('--reason',required=True)
    p.add_argument('--receipt',default='artifacts/final/selection.json')
    args=p.parse_args()
    config=torch.load(args.checkpoint,map_location='cpu',weights_only=False)['config']
    freeze_selection(args.checkpoint,config,args.validation_report,args.reason,args.receipt)
    print(args.receipt)
