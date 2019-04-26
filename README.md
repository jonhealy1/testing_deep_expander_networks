# testing deep expander networks on ResNet, WideResNet, and PyramidNet (ResNet version)

Notes: I added these models to the assignment 7 code but on every second or third training run 
I was getting weird results (~10% instead of the expected ~80%). I had to port the models 
over to new training code as I was wanting to train for 5+ hours per model. I couldn't
figure out what the problem was with the assignment 7 code - it worked fine when I was
doing assignment 7, so obviously I messed something up somewhere. I thought that if I tested
all of these things I could gain some insight and develop my own model. I did not 
develop my own code from scratch for deep expander networks. I had worked with the 
code previously and I felt that writing my own version was going to just be
basically copying. I thought that I could do something more interesting.
I thought that the results from their paper were interesting and I was wondering why
their paper wasn't receiving more attention. I would have liked to work more on this
project. There was a lot of coding that went into this project. Some of the code that I 
borrowed did not work as expected and there is quite a lot of code that did not make 
it to this repository. 


ResNet/ Wide ResNet and Training Code:

https://github.com/meliketoy/wide-resnet.pytorch

ResNet:

python main.py --lr 0.025 --net_type resnet --depth 50 --dropout 0.3 --dataset cifar10

Wide-ResNet:

python main.py --lr 0.025 --net_type wide-resnet --depth 16 --widen_factor 10 --dropout 0.3 --dataset cifar10

PyramidNet:

https://github.com/dyhan0920/PyramidNet-PyTorch

python main.py --lr 0.025 --net_type pyramid-resnet --depth 38 --dropout 0.3 --dataset cifar10

Deep Expander Networks:

https://github.com/drimpossible/Deep-Expander-Networks

python main.py --lr 0.025 --net_type resnet-deep-expand --depth 50 --dropout 0.3 --dataset cifar10

python main.py --lr 0.025 --net_type wide-resnet-deep-expander --depth 16 --widen_factor 10 --dropout 0.3 --dataset cifar10 

python main.py --lr 0.025 --net_type pyramid-deep-expand --depth 38 --dropout 0.3 --dataset cifar10

