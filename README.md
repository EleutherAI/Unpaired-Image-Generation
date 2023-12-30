# Unpaired-Image-Generation

## Setup

### Environment
The environment can be imported from `environment.yaml`.

### Datasets
No setup should be required to load CIFAR-10 or CIFAR-100. To use COCO, ensure that the image and annotation paths agree with those in `trainer.py`.

### Training a model
`python -m model.trainer --config default --debug`

The default.yml config should have all necessary information to create your own configs. Configs and args can be changed in `model/config_utils.py`