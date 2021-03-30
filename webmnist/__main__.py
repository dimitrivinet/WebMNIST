
import argparse
import os

from webmnist.train import train
from webmnist.export import export

parser = argparse.ArgumentParser(description='Train on MNIST')

parser.add_argument('-o', '--output-path', 
    type=str, 
    default="./trained_models", 
    help="path to save models to (train), path to export model to (export)",
)
parser.add_argument('-i', '--input-path', 
    type=str, 
    default="./trained_models/best.pt", 
    help="path to trained model",
)
parser.add_argument('-e', '--num-epochs', 
    type=int, 
    default=3, 
    help="number of epochs to train for",
)
parser.add_argument('-a', '--save-all', 
    action='store_true', 
    help="save all checkpoints",
)
parser.add_argument('--train', action='store_true', help="train model")
parser.add_argument('--export', action='store_true', help="export model")
args = parser.parse_args()
# print(args)

TRAIN = args.train
EXPORT = args.export
OUTPUT_PATH = args.output_path
INPUT_PATH = args.input_path
NUM_EPOCHS = args.num_epochs
SAVE_ALL = args.save_all


if TRAIN:
    if not os.path.exists(OUTPUT_PATH):
        print(f"crating dir {OUTPUT_PATH}")
        os.mkdir(OUTPUT_PATH)

    train(OUTPUT_PATH, SAVE_ALL, NUM_EPOCHS)
    
elif EXPORT:
    if os.path.exists(INPUT_PATH):       

        export(INPUT_PATH, OUTPUT_PATH)