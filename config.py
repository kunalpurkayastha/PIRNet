import argparse

def get_args():
    parser = argparse.ArgumentParser(description='MRI Scan Analysis Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--data_dir', type=str, default='./data', help='directory of the dataset')
    parser.add_argument('--output_dir', type=str, default='./output', help='directory to save outputs')
    return parser.parse_args()