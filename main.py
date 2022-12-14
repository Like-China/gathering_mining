import os
import torch
import warnings
from trainer.train import train
from settings import set_args
from data.porto.porto1500350.evaluator.evaluate import evaluate
os.environ['CUDA_ENABLE_DEVICES'] = '0'
os.environ["CUDA_LAUNCH_BLOCKICNG"] = '1'
torch.backends.cudnn.enabled=False
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # 0 for training, otherwise validation
    train_or_evaluate = 0
    args = set_args()
    if train_or_evaluate == 0:
        train(args)
    else:
        evaluate(args)

