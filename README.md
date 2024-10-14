# Universal Super Resolution Neural Operator

This repository contains the official implementation for USRNO.
The code is based on Ubuntu 18.04, pytorch 2.1.1+cu121.

## Quick Start

Clone this repo with:

```cmd
git clone https://github.com/ZXS-Labs/USRNO.git
```

Get the environment using pip:

```cmd
pip install -r requirements.txt
```

We show all the packages of our environment in `requirements.txt`, and you might not need to install all of these.

## Train

`python train.py --config configs/train_edsr-sronet.yaml`
if you want to change the model, please check the config file in `./configs` and feel free to change the yaml files.

For example:

```yaml
model:
  name: sronet
  args:
    encoder_spec:
      name: edsr-baseline ## or rdn, rdn-3d
      args:
        no_upsampling: true
        #feature_dim : 64
    width: 256 # 512
    blocks: 16
```

## Test

Check the config files in `./configs` for all testing configurations.

`python test.py --config configs/test_srno.yaml --model your_model_path.pth --mcell True`

## Demo

`python demo.py --input input.png --model your_model_path.pth --scale 2 --output output.png`

## Acknowledgements

This code is built on [SRNO](https://github.com/2y7c3/Super-Resolution-Neural-Operator) , [LIIF](https://github.com/yinboc/liif) and [LTE](https://github.com/jaewon-lee-b/lte)

