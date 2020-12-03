import argparse

def options():
    parser = argparse.ArgumentParser()

    ## Data
    parser.add_argument("--data_path", help="train data path", \
        default="../datasets/data/data.npy", type=str)
    parser.add_argument("--label_path", help="label data path", \
        default="../datasets/data/arousal_labels.npy", type=str)
    
    
    ## NetWork
    parser.add_argument("--num_epochs", help="number of epochs", \
        default=10, type=int)
    parser.add_argument("--learning_rate", help="learning_rate", \
        default=1e-3, type=float)
    parser.add_argument("--batch_size", help="batch size", \
        default=1, type=int)
    parser.add_argument("--checkpoint_path", help="checkpoint path", \
        default="../weights/checkpoint.pth", type=str)

    args = parser.parse_args()
    return args

opt = options()