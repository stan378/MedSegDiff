import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def normalize(arr):
    arr = arr.astype('float')
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def display_histogram(img):
  histogram = img.histogram()
  for i in range(0, 256):
    plt.bar(i, histogram[i], color='black', alpha=0.3)
  plt.ylim([0, 100])
  plt.show()

data_path = "D:/Segmentacja/Experiments/data/mri/"
normalized_path = "D:/Segmentacja/Experiments/data/mri_normalized/"
folders = os.listdir(data_path)

for folder in folders:
  patient_path = data_path + folder
  images = os.listdir(patient_path)

  for image in images:
    img_path = patient_path + "/" + image
    img = Image.open(img_path)
    if 'mask' in image:
      if not os.path.exists(normalized_path + folder):
        os.makedirs(normalized_path + folder)
      img.save(normalized_path + folder + "/" + image)
      continue

    new_img = Image.fromarray(normalize(np.array(img)).astype('uint8'))

    if not os.path.exists(normalized_path + folder):
      os.makedirs(normalized_path + folder)

    new_img.save(normalized_path + folder + "/" + image)
print("Ok!")