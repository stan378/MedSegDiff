import os
import argparse
import random
import math
from distutils.dir_util import copy_tree

parser = argparse.ArgumentParser(description="Build and validate model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--test_size", type=float, help="Size of test set")
parser.add_argument("-i", "--input", help="Path to data folder")

args = parser.parse_args()
config = vars(args)
all_patients = os.listdir(config['input'])
patients = os.listdir(config['input'])
test_size = int(math.floor(len(patients) * config['test_size']))
groups = {}
ind = 1

while len(patients) != 0:
  test_patients = random.sample(patients, k=test_size)
  groups[ind] = test_patients
  patients = list(set(patients) - set(test_patients))
  ind = ind + 1

for key in groups:
  test_patients = groups[key]
  for test_patient in test_patients:
    src = os.path.join(config['input'], test_patient)
    dst = os.path.join(config['input'], "..", f"split{key}", "test", test_patient)
    if not os.path.exists(dst):
      os.makedirs(dst)
    copy_tree(src, dst)

  train_patients = list(set(all_patients) - set(test_patients))
  for train_patient in train_patients:
    src = os.path.join(config['input'], train_patient)
    dst = os.path.join(config['input'], "..", f"split{key}", "train", train_patient)
    if not os.path.exists(dst):
      os.makedirs(dst)
    copy_tree(src, dst)

print("Ok!")