import argparse

def options():
    parser = argparse.ArgumentParser()

    ## Data Parameters
    parser.add_argument("--data_path", help="train data path", \
        default="./data/small_data.npy", type=str)
    parser.add_argument("--label_path", help="label data path", \
        default="./data/arousal_labels.npy", type=str)
    parser.add_argument("--test_size", help="test data size: test_size x len(data)", \
        default=0.2, type=float)
    parser.add_argument("--val_size", help="validate data size: test_size x len(data)", \
        default=0.1, type=float)
    parser.add_argument("--seed", help="random seed", \
        default=42, type=int)
    parser.add_argument("--mode", help="split the data 1 or 2", \
        default=0, type=int)

    parser.add_argument("--train_data_path", help="train data path", \
        default="./datasets/data/train_data.npy", type=str)
    parser.add_argument("--train_label_path", help="train label path", \
        default="./datasets/data/train_label.npy", type=str)

    parser.add_argument("--test_data_path", help="test data path", \
        default="./datasets/data/test_data.npy", type=str)
    parser.add_argument("--test_label_path", help="test label path", \
        default="./datasets/data/test_label.npy", type=str)

    parser.add_argument("--shuffle", help="wheate shuffle data", \
        default=False, type=bool)
    
    
    ## Training Parametes 
    parser.add_argument("--test", help="test the model", \
        default=False, type=bool)
    parser.add_argument("--model", help="model name {basemodel, EEGNet, TSception}", \
        default='basemodel', type=str)
    parser.add_argument("--small", help="use small data to train", \
        default=False, type=bool)
    parser.add_argument("--num_epochs", help="number of epochs", \
        default=10, type=int)
    parser.add_argument("--learning_rate", help="learning rate", \
        default=1e-3, type=float)
    parser.add_argument("--batch_size", help="batch size", \
        default=256, type=int)
    parser.add_argument("--weight_decay", help="weight decay", \
        default=1e-4, type=float)
    parser.add_argument("--checkpoint_dir", help="checkpoints dir", \
        default="./weights/", type=str)
    parser.add_argument("--checkpoint_path", help="checkpoint path", \
        default="./weights/checkpoint.pth", type=str)
    parser.add_argument("--pretrained", help="wheather to load checkpoint",\
        default=False, type=bool)
    parser.add_argument("--normalized", help="wheather to normalize network paramters", \
        default=True, type=bool)
    parser.add_argument("--patient", help="early stop threshold", \
        default=20, type=int)


    ## Test Pramaters
    parser.add_argument("--best", help="wheater to use the best state", \
        default=False, type=bool)

    
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
        default=0.25, type=float)
    parser.add_argument("--lambda_", help="regulization network paramters", \
        default=1e-6, type=float)


    ## Visualize Parameters
    parser.add_argument('--display_port', type=int, default=8888, \
         help='display port')


    args = parser.parse_args()
    return args

opt = options()