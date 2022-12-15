from fetch import fetch
import numpy as np
import os
import sys
import time



def load(img):
  # preprocess

  aspect_ratio = img.size[0]/img.size[1]
  img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))

  img = np.array(img)
  ydim,xdim=(np.asarray(img.shape)[:2]-224)//2
  retimg = img = img[ydim:ydim+224, xdim:xdim+224]
  #print(img)

  import matplotlib.pyplot as plt
  plt.imshow(img)
  plt.show()



if __name__ == "__main__":

  import ast
  lbls = fetch("https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt")
  lbls = ast.literal_eval(lbls.decode('utf-8'))

  #print(lbls)

  arg = sys.argv[1]
  from PIL import Image
  img = Image.open(arg)

  load(img)

