import argparse
import os
os.environ['CUDA_ENABLE_DEVICES'] = '0'
os.environ["CUDA_LAUNCH_BLOCKICNG"] = '1'
import torch
torch.backends.cudnn.enabled=False
import warnings
warnings.filterwarnings("ignore")
import constants

def setArgs():
    cityname = constants.cityname
    scale = constants.scale
    time_size = constants.time_size
    
    
    
   
    parser = argparse.ArgumentParser(description="train.py")
    '''
    *************************************************************************
    训练参数
    *****************
    '''
    parser.add_argument("-max_num_line", type=int, default=80000, help='读取训练集大小')
    parser.add_argument("-read_val_nums", type=int, default=10000, help="读取的验证集大小")
    parser.add_argument("-iter_num", default=3000000,help="总的训练迭代次数")
    parser.add_argument("-max_length", default=200, help="The maximum length of the target sequence")
    parser.add_argument("-c_method", default=1, help="1 k-means 2-DBSCAN 3-Hire")
    
    '''
    *************************************************************************
     Region参数
    *************************************************************************
    '''
    parser.add_argument("-cityname", default = cityname, help="city name")
    parser.add_argument("-scale", type=float, default= scale, help="city scale")
    parser.add_argument("-time_size", type=float, default= time_size, help="time span nums")
    parser.add_argument("-data", default= os.path.join(os.getcwd(),'data',cityname, cityname+str(int(scale*100000))+str(time_size)), help="训练集和模型存储目录")
    parser.add_argument("-checkpoint", default= os.path.join(os.getcwd(),'data',cityname, cityname+str(int(scale*100000))+str(time_size),'checkpoint.pt'), help="checkpoint存放目录")
    '''
    *************************************************************************
    神经网络层参数
    *************************************************************************
    '''
    parser.add_argument("-prefix", default="exp", help="Prefix of trjfile")
    
    parser.add_argument("-num_layers", type=int, default=3, help="Number of layers in the RNN cell")
    parser.add_argument("-bidirectional", type=bool, default=True, help="True if use bidirectional rnn in encoder")
    parser.add_argument("-hidden_size", type=int, default=32, help="The hidden state size in the RNN cell")
    parser.add_argument("-embedding_size", type=int, default=256, help="The word (cell) embedding size")
    parser.add_argument("-dropout", type=float, default=0.2, help="The dropout probability")
    parser.add_argument("-max_grad_norm", type=float, default=5.0, help="The maximum gradient norm")
    parser.add_argument("-learning_rate", type=float, default=0.0001)
    parser.add_argument("-batch", type=int, default=constants.train_batch, help="The batch size") 
    parser.add_argument("-max_apn_num", type=int, default=1200, help="The batch size") 
    parser.add_argument("-generator_batch", type=int, default=1024, help="The maximum number of words to generate each time.The higher value, the more memory requires.")
    parser.add_argument("-t2vec_batch", type=int, default=2048, help="""验证集每次编码的batch_size""")
    parser.add_argument("-start_iteration", type=int, default=0)
    parser.add_argument("-loss1_use_freq", type=int, default=100)
    parser.add_argument("-epochs", type=int, default=15, help="The number of training epochs")
    parser.add_argument("-print_freq", type=int, default=1, help="Print frequency")
    parser.add_argument("-save_freq", type=int, default=1, help="Save frequency")
    parser.add_argument("-cuda", type=bool, default=True, help="True if we use GPU to train the model")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    from train import train
    args = setArgs()
    train(args)
    
