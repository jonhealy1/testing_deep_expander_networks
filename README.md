# resnet-variations
Testing different versions of resnet for school.

ResNet/ Wide ResNet and Training Code:

https://github.com/meliketoy/wide-resnet.pytorch

ResNet:

python main.py --lr 0.025 --net_type resnet --depth 50 --widen_factor 10 --dropout 0.3 --dataset cifar10

Wide-ResNet:

python main.py --lr 0.025 --net_type wide-resnet --depth 16 --widen_factor 10 --dropout 0.3 --dataset cifar10

PyramidNet:

https://github.com/dyhan0920/PyramidNet-PyTorch

python main.py --lr 0.025 --net_type pyramid-resnet --depth 38 --dataset cifar10

Deep Expander Networks:

https://github.com/drimpossible/Deep-Expander-Networks

python main.py --lr 0.025 --net_type resnet-deep-expand --depth 50 --widen_factor 10 --dropout 0.3 --dataset cifar10

python main.py --lr 0.1 --net_type wide-resnet-deep-expander --depth 28 --widen_factor 10 --dropout 0.3 --dataset cifar10 

python main.py --lr 0.025 --net_type pyramid-deep-expand --depth 38 --dataset cifar10

