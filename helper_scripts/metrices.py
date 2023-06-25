import sys
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import mean_squared_error
import os

def iou(groundtruth_mask, pred_mask):
  intersect = np.sum(pred_mask * groundtruth_mask)
  union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
  iou = np.mean(intersect / union)
  return round(iou, 3)

def convertPredicted(img, threshold = 200):
  for j in range(img.shape[0]):
    for i in range(img.shape[1]):
      if img[j][i] > threshold:
        img[j][i] = 255
      else:
        img[j][i]= 0
  return img

def change_color(img, color):
  data = np.array(img)  # "data" is a height x width x 4 numpy array
  red, green, blue = data.T  # Temporarily unpack the bands for readability

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

  data[..., :][white_areas.T] = (new_red, new_green, new_blue)  # Transpose back needed
  im2 = Image.fromarray(data)
  return im2

def mergeMasks(target, pred, threshold = 200):
  pred = Image.open(pred).convert('L')
  pred = Image.fromarray(convertPredicted(np.array(pred), threshold)).convert('1')
  target = Image.open(target).convert('1')
  image = Image.new('1', (256, 256), 1)

  new_img = ImageChops.logical_and(pred, target)
  pred_mask = ImageChops.logical_and(pred, image)
  target = target.convert('RGB')
  pred = pred.convert('RGB')

  pred = change_color(pred, 'red')
  target = change_color(target, 'green')

  target.paste(pred, (0, 0), pred_mask)
  target.paste(Image.new('RGB', (256, 256), (255,255,255)), (0,0), new_img)
  return target

def precision_score_(groundtruth_mask, pred_mask):
  intersect = np.sum(pred_mask * groundtruth_mask)
  total_pixel_pred = np.sum(pred_mask)
  if total_pixel_pred == 0:
    return 0
  precision = np.mean(intersect / total_pixel_pred)
  return round(precision, 3)

def recall_score_(groundtruth_mask, pred_mask):
  intersect = np.sum(pred_mask * groundtruth_mask)
  total_pixel_truth = np.sum(groundtruth_mask)
  recall = np.mean(intersect / total_pixel_truth)
  return round(recall, 3)

def accuracy(groundtruth_mask, pred_mask):
  intersect = np.sum(pred_mask * groundtruth_mask)
  union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
  xor = np.sum(groundtruth_mask == pred_mask)
  acc = np.mean(xor / (union + xor - intersect))
  return round(acc, 3)

def dice_coef(groundtruth_mask, pred_mask):
  intersect = np.sum(pred_mask * groundtruth_mask)
  total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
  dice = np.mean(2 * intersect / total_sum)
  return round(dice, 3)

def calculate_metrics(mask, pred):
  pred = Image.open(pred).convert('L')
  pred = Image.fromarray(convertPredicted(np.array(pred))).convert('1')
  target = Image.open(mask).convert('1')

  target = np.array(target)
  pred = np.array(pred)

  mse = round(mean_squared_error(target.astype(int) , pred.astype(int)),5)
  iou_value = iou(target, pred)
  precision = precision_score_(target, pred)
  recall = recall_score_(target, pred)
  accuracy_value = accuracy(target, pred)
  dice_value = dice_coef(target, pred)
  return iou_value, precision, recall, accuracy_value, dice_value, mse

def display_histogram(img):
  histogram = img.histogram()
  print(np.asarray(histogram>0.98).nonzero())
  print(len(conv.histogram()))
  for i in range(0, 256):
    plt.bar(i, histogram[i], color='black', alpha=0.3)
  plt.ylim([0, 100])
  plt.show()

#pred = Image.open(sys.argv[1])
#target = Image.open(sys.argv[2])
result_path = f'D:/Segmentacja/Experiments/Grayscale/results330000/'
test_folder = f'D:/Segmentacja/Experiments/test/'

patients = os.listdir(result_path)
iou_avg = 0
precision_avg = 0
recall_avg = 0
accuracy_avg = 0
dice_value_avg = 0
mse_avg = 0
number_of_cases = 0
for patient in patients:
  pred_folder = result_path + patient
  target_folder = test_folder + patient
  mergedFolder = pred_folder + '/Merged/'
  convertedFolder = pred_folder + "/Converted/"

  if not os.path.exists(mergedFolder):
    os.makedirs(mergedFolder)

  if not os.path.exists(convertedFolder):
    os.makedirs(convertedFolder)

  preds = os.listdir(pred_folder)
  for pred in preds:
    if 'Convert' in pred or 'Merged' in pred:
      continue

    print(pred)
    # Converted images
    conv = Image.open(pred_folder + '/' + pred).convert('L')
    histogram = conv.histogram()
    histogram_cumsum = np.cumsum(histogram) / np.sum(histogram)
    threshold = np.asarray(histogram_cumsum > 0.98).nonzero()[0][0]
    print(f'Threshold: {threshold}')
    conv = Image.open(pred_folder + '/' + pred).convert('L')
    conv = Image.fromarray(convertPredicted(np.array(conv), threshold)).convert('1')
    conv.save(convertedFolder + pred + '.jpg')

    # Merged images
    mask = '/' + pred[:-15] + '_mask.tif'
    mask_path = target_folder + mask
    pred_path = pred_folder + '/' + pred
    mergeMasks(mask_path, pred_path, threshold).save(mergedFolder + pred[:-15] + '_merged' + '.png')

    iou_value, precision, recall, accuracy_value, dice_value, mse = calculate_metrics(mask_path, pred_path)
    print(f'\tIOU: {iou_value}\n\tPrecision: {precision}\n\tRecall: {recall}\n\t'
          f'Accuracy: {accuracy_value}\n\tDice coef: {dice_value}\n\tMSE: {mse}')

    iou_avg = iou_avg + iou_value
    precision_avg = precision_avg + precision
    recall_avg = recall_avg + recall
    accuracy_avg = accuracy_avg + accuracy_value
    dice_value_avg = dice_value_avg + dice_value
    mse_avg = mse_avg + mse
    number_of_cases = number_of_cases + 1

print('Average:')
print(f'\tIOU: {iou_avg / number_of_cases}\n\tPrecision: {precision_avg / number_of_cases}\n\tRecall: {recall_avg / number_of_cases}\n\t'
          f'Accuracy: {accuracy_avg / number_of_cases}\n\tDice coef: {dice_value_avg / number_of_cases}\n\tMSE: {mse_avg / number_of_cases}')