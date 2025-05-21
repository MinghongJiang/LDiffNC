# 基于diffusion模型的不平衡分类方法研究

# LDiffNC (Latent-based Diffusion Model with Neural Collapse)

## Requirements:

All codes are written by Python 3.9 with 

```
numpy 1.26.4
scikit-learn 1.6.1
denoising_diffusion_pytorch 2.2.1
ema_pytorch 0.7.7
accelerate 1.3.0
einops 0.8.0
matplotlib 3.9.4
scipy 1.13.1
torch 2.7.0
torch.version.cuda 12.6
torchvision 0.22.0

```

## Training

To train the model(s), run this command:

python main.py --datapath ./data/... --model_fixed ./pretrained_models/...

Example:
python main.py --datapath ./data/CIFAR100_LT001 --model_fixed ./pretrained_models/resnet32_cifar100_lt001.checkpoint

## Evaluation

To evaluate model, run:

python main.py --datapath ./data/... --model_fixed ./pretrained_models/... --eval .\saved_models\...

Example:
python main.py --datapath ./data/CIFAR100_LT001 --model_fixed ./pretrained_models/resnet32_cifar100_lt001.checkpoint --eval .\saved_models\ckpt_best_ce.checkpoint   

## Acknowledgement

[ETF-DR](https://github.com/NeuralCollapseApplications/ImbalancedLearning)

[INC](https://github.com/Pepper-lll/NCfeature)

[LDMLR](https://github.com/AlvinHan123/LDMLR)


