import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_info',
                        type=str,
                        help='info that will be displayed when logging',
                        default='CNN')

    parser.add_argument('--lr',
                        type=float,
                        help='learning rate',
                        default=1e-4)

    parser.add_argument('--weight_decay',
                        type=float,
                        help='L2 weight decay',
                        default=1e-5)


    parser.add_argument('--num_workers',
                        type=int,
                        help='number of subprocesses for dataloader',
                        default=16)

    parser.add_argument('--gpu',
                        type=str,
                        help='id of gpu device(s) to be used',
                        default='0')

    parser.add_argument('--train_batch_size',
                        type=int,
                        help='batch size for training phase',
                        default=8)
    
    parser.add_argument('--val_batch_size',
                        type=int,
                        help='batch size for valing phase',
                        default=8)

    parser.add_argument('--test_batch_size',
                        type=int,
                        help='batch size for test phase',
                        default=32)

    parser.add_argument('--num_epochs',
                        type=int,
                        help='number of training epochs',
                        default=50)
    
    parser.add_argument('--log_file',
                        type=str,
                        help='path to log file',
                        default='exp/CNN.log')

    return parser