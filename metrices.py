import sys
#from torchmetrics import JaccardIndex
from PIL import Image, ImageChops
import numpy as np

#pred = Image.open(sys.argv[1])
#target = Image.open(sys.argv[2])
pred = "D:/Segmentacja/TCGA_HT_8113_19930809_18_output_ens.jpg"
target = "D:/Segmentacja/TCGA_HT_8113_19930809_18_mask.tif"

def iou(groundtruth_mask, pred_mask):
  intersect = np.sum(pred_mask * groundtruth_mask)
  union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
  iou = np.mean(intersect / union)
  return round(iou, 3)

def convertPredicted(img):
  for j in range(img.shape[0]):
    for i in range(img.shape[1]):
      if img[j][i] > 200:
        img[j][i] = 255
      else:
        img[j][i]= 0
  return img

def change_color(img, color, trans):
  data = np.array(img)  # "data" is a height x width x 4 numpy array
  red, green, blue, alpha = data.T  # Temporarily unpack the bands for readability

  white_areas = (red == 255) & (blue == 255) & (green == 255)

  new_red = 0
  new_green = 0
  new_blue = 0
  if color == 'red':
    new_red = 255
  elif color == 'green':
    new_green = 255
  elif color == 'blue':
    new_blue = 255

  data[..., :-1][white_areas.T] = (new_red, new_green, new_blue)  # Transpose back needed
  data[...,-1] = trans
  im2 = Image.fromarray(data)
  return im2

def mergeMasks(target, pred):
  pred = Image.open(pred).convert('L')
  pred = Image.fromarray(convertPredicted(np.array(pred))).convert('1')
  target = Image.open(target).convert('1')

  new_img = ImageChops.logical_and(pred, target).convert('RGBA')
  target = target.convert('RGBA')
  pred = pred.convert('RGBA')

  pred = change_color(pred, 'red', 64)
  target = change_color(target, 'green', 64)

  new_img.paste(target, (0,0), target)
  new_img.paste(pred, (0,0), pred)
  return new_img

mergeMasks(target, pred).show()
#target = Image.open(target).convert('RGBA')
#change_color(target, 'red', 128)

#img2 = Image.fromarray(convertPredicted(np.array(pred)))
#img2.show()
#print(np.array(target).shape)
#print(np.array(pred).shape)

#print(iou(np.array(target), convertPredicted(np.array(pred))))

#im3 = ImageChops.logical_and(target, img2)
#img2 = Image.fromarray(convertPredicted(np.array(pred)))
#img3 = Image.fromarray(np.array(target))

#img2.show()
#img3.show()

