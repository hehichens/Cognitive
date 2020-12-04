import argparse

def options():
    parser = argparse.ArgumentParser()

    ## Data Parameters
    parser.add_argument("--data_path", help="train data path", \
        default="./datasets/data/data.npy", type=str)
    parser.add_argument("--label_path", help="label data path", \
        default="./datasets/data/arousal_labels.npy", type=str)
    
    
    ## Training Parametes 
    parser.add_argument("--test", help="test the model", \
        default=False, type=bool)
    parser.add_argument("--model", help="model name {basemodel, EEGNet, TSception}", \
        default='basemodel', type=str)
    parser.add_argument("--num_epochs", help="number of epochs", \
        default=10, type=int)
    parser.add_argument("--learning_rate", help="learning_rate", \
        default=1e-3, type=float)
    parser.add_argument("--batch_size", help="batch size", \
        default=256, type=int)
    parser.add_argument("--checkpoints_dir", help="checkpoints dir", \
        default="../weights/", type=str)
    parser.add_argument("--checkpoint_path", help="checkpoint path", \
        default="../weights/checkpoint.pth", type=str)

    
    ## NetWork Parameters
    parser.add_argument("--num_class", help="data classfication numbers", \
        default=2, type=int)
    parser.add_argument("--num_channel", help="data channels size", \
        default=40, type=int)
    parser.add_argument("--num_dim", help="data dimensions", \
        default=101, type=int)
    parser.add_argument("--sampling_rate", help="sampling rate", \
        default=128, type=int)
    parser.add_argument("--num_T", help="T num", \
        default=9, type=int)
    parser.add_argument("--num_S", help="S num", \
        default=6, type=int)
    parser.add_argument("--hidden_size", help="hidden layer size", \
        default=128, type=int)
    parser.add_argument("--dropout_rate", help="dropout rate", \
        default=0.2, type=float)


    ## Visualize Parameters
    parser.add_argument('--display_port', type=int, default=8888, \
         help='display port')


    args = parser.parse_args()
    return args

opt = options()