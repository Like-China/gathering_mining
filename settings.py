import argparse
import os


# Constant parameters
PAD = 0
BOS = 1
EOS = 2
UNK = 3  # Low frequency word encoding
START = 4  # Start token coding at 4
# beijing
lons_range_bj = [116.25, 116.55]
lats_range_bj = [39.83, 40.03]
# porto
lons_range_pt = [-8.735, -8.156]
lats_range_pt = [40.953,  41.307]
# The ratio of training set 
train_ratio, val_ratio,  test_ratio= 0.6, 0.2, 0.2


# variable parameters
min_len, max_len = 20, 100
city = "porto"
scale = 0.0001
time_size = 400
min_lifetime = 3
min_group_trj_nums = 2
each_num = 5  # The number of t_loss was calculated for each group
n = 30  # kmeans cluster number
# The number of training sets and test sets each time
train_batch = 5000
val_batch = 5000
# SCAN参数
ep = 0.6


def set_args():
    parser = argparse.ArgumentParser(description="train.py")
    '''
    *************************************************************************
    parameters of training
    *****************
    '''
    parser.add_argument("-max_train_num", type=int, default=1000000, help='Size of the training set')
    parser.add_argument("-max_val_nums", type=int, default=10000, help="Size of the validation set")
    parser.add_argument("-iter_num", default=3000000, help="Total number of training iterations")
    parser.add_argument("-max_length", default=max_len, help="The maximum length of the target sequence")
    parser.add_argument("-c_method", default=0, help="0 SCAN 1 k-means 2-DBSCAN 3-Hire")
    parser.add_argument("-max_invalid_num", default=10, help="Terminating of the training if training losses are not reduced by more than that")

    '''
    *************************************************************************
     parameters of Region
    *************************************************************************
    '''
    parser.add_argument("-city", default=city, help="city name")
    parser.add_argument("-scale", type=float, default=scale, help="city scale")
    parser.add_argument("-time_size", type=float, default=time_size, help="time span nums")
    parser.add_argument("-data", default=os.path.join('./data', city,
                                                      city + str(int(scale * 100000)) + str(time_size)),
                        help="Training set and model store directory")
    parser.add_argument("-checkpoint", default=os.path.join('./data', city,
                                                            city + str(int(scale * 100000)) + str(time_size),
                                                            'checkpoint.pt'), help="checkpoint directory")
    '''
    *************************************************************************
    Neural network layer parameters
    *************************************************************************
    '''
    parser.add_argument("-prefix", default="exp", help="Prefix of trjfile")

    parser.add_argument("-num_layers", type=int, default=8, help="Number of layers in the RNN cell")
    parser.add_argument("-bidirectional", type=bool, default=True, help="True if use bidirectional rnn in encoder")
    parser.add_argument("-hidden_size", type=int, default=16, help="The hidden state size in the RNN cell")
    parser.add_argument("-embedding_size", type=int, default=16, help="The word (cell) embedding size")
    parser.add_argument("-dropout", type=float, default=0.2, help="The dropout probability")
    parser.add_argument("-max_grad_norm", type=float, default=5.0, help="The maximum gradient norm")
    parser.add_argument("-learning_rate", type=float, default=0.0001)
    parser.add_argument("-batch", type=int, default=train_batch, help="The batch size")
    parser.add_argument("-max_apn_num", type=int, default=400, help="The batch size")
    parser.add_argument("-generator_batch", type=int, default=1024,
                        help="The number of words to generate each time.The higher value, the more memory requires.")
    parser.add_argument("-t2vec_batch", type=int, default=1024, help="""The batch_size of validation set encodeing every time""")
    parser.add_argument("-start_iteration", type=int, default=0)
    parser.add_argument("-loss1_use_freq", type=int, default=100)
    parser.add_argument("-epochs", type=int, default=15, help="The number of training epochs")
    parser.add_argument("-print_freq", type=int, default=2, help="Print frequency")
    parser.add_argument("-save_freq", type=int, default=4, help="Save frequency")
    parser.add_argument("-cuda", type=bool, default=True, help="True if we use GPU to train the model")

    args = parser.parse_args()
    return args


