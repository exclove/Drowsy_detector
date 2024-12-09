import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Global Namespace for Hyperparameters")

    # Hyperparameters
    parser.add_argument('--in_channel', type=int, default=1, help='Number of input channels')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--load_model', type=bool, default=False, help='Load model')
    parser.add_argument('--gamma', type=float, default=0.25, help='Gamma for tadam')
    parser.add_argument('--total_steps_num', type=int, default=60_000, help='Total_steps for tadam')

    # # Dataset settings
    # parser.add_argument('--csv_file', type=str, default='cat_dog_labels.csv', help='Path to the CSV file with labels')
    # parser.add_argument('--root_dir', type=str, default='cats_and_dogs', help='Root directory of the dataset')

    args = parser.parse_args()
    return args