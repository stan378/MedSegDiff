import scipy.ndimage as ndi
from PIL import Image
import numpy as np

img = Image.open('D:/Segmentacja/Experiments/data/mri_normalized/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_11.tif')
im_s1 = ndi.gaussian_filter(img, sigma=1)

img2 = Image.fromarray(im_s1)
img.show()
img2.show()

