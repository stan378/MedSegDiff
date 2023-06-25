import os
import argparse

parser = argparse.ArgumentParser(description="Build and validate model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--test_size", type=float, help="Size of test set")
parser.add_argument("-k", "--kfold", type=int, help="Kfold")
parser.add_argument("-i", "--input", help="Path to data folder")

args = parser.parse_args()
config = vars(args)
print(type(config['kfold']))

#path = "D:/Segmentacja/Experiments/data/mri_normalized"
#patients = os.listdir(path)



print("Ok!")
		