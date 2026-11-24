import argparse
from .common import load_config
from .engine import train_run

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--variant');p.add_argument('--seed',type=int);p.add_argument('--output');p.add_argument('--resume',action='store_true')
    a=p.parse_args();c=load_config(a.config)
    if a.variant:c['variant']=a.variant
    if a.seed is not None:c['seed']=a.seed
    if a.output:c['output_dir']=a.output
    if a.resume:c['resume']=True
    train_run(c)
if __name__=='__main__':main()
