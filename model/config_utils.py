import argparse
import os
import yaml
import argparse
from types import SimpleNamespace

def load_config(config_name):
    config_path = os.path.join('./config', config_name + '.yml')

    with open(config_path, 'r') as stream:
        data = yaml.safe_load(stream)

    data_obj = SimpleNamespace(**data)
    data_obj.CONFIG_NAME = config_name
    return data_obj

def parse_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', type=str, default='default')
    parser.add_argument('--model', type=str, default='default')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sample_latent', action='store_true')
    parser.add_argument('--t2i', action='store_true')
    parser.add_argument('--i2t', action='store_true')
    parser.add_argument('--reconstruction', action='store_true') # img reconstruction for debugging
    args = parser.parse_args()
    return load_config(args.config), args