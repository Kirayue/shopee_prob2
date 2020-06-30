from run import run
from config import get_cfg_defaults
from argparse import ArgumentParser

import torch.backends.cudnn as cudnn
# Enable auto-tuner to find the best algorithm to use for your hardware.
cudnn.benchmark = True


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', "--mode", type=str, default='train', choices=['train', 'infer', 'eval'], help='Model for training or inferencing')
    parser.add_argument('-c', "--config", type=str, default='Experiment.yaml', help='config file for experiment')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()
    run(args.mode, cfg)
