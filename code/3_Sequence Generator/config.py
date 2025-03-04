import argparse

def set_args():
    """
    Define and parse command-line arguments for the text generation model.
    """
    # Create an argument parser with a description
    parser = argparse.ArgumentParser(description="Test generation")

    # Define command-line arguments
    parser.add_argument('--batch_size', default=12, type=int,
                        help='Batch size for training')
    parser.add_argument('--pretrain_model_path', default='./t5pretrain', type=str,
                        help='Path to the pre-trained model')
    parser.add_argument('--data_path', default='../../data/5_merged_result_4.json', type=str,
                        help='Path to the processed training data')
    parser.add_argument('--epochs', default=20, type=int, required=False,
                        help='Number of training epochs')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--learning_rate', default=2e-4, type=float, required=False,
                        help='Learning rate for the optimizer')
    parser.add_argument('--output_dir', default='./output', type=str, required=False,
                        help='Directory to save output results')
    parser.add_argument('--warmup_steps_rate', default=0.005, type=float, required=False,
                        help='Proportion of total steps used for warm-up')

    return parser.parse_args()  # Parse and return the arguments
