import argparse
from .common import load_config
from .scheduling import Policy
from .engine import evaluate_run,simple_baselines

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--dataset-path');p.add_argument('--simple-baselines',action='store_true');p.add_argument('--split',default='val',choices=['val','test']);p.add_argument('--output',required=True);p.add_argument('--policy',default='full');p.add_argument('--policy-config');p.add_argument('--device',default='auto');p.add_argument('--batch-size',type=int,default=16);p.add_argument('--smoke-limit',type=int)
    a=p.parse_args()
    if a.simple_baselines:simple_baselines(a.dataset_path,a.output,a.split);return
    if not a.checkpoint:p.error('--checkpoint required')
    policy=Policy(**load_config(a.policy_config)) if a.policy_config else Policy(kind=a.policy)
    evaluate_run(a.checkpoint,a.split,a.output,policy,a.device,a.batch_size,a.smoke_limit)
if __name__=='__main__':main()
