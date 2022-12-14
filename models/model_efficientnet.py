import torch
from torch import nn
import math

#(Inverted Residuals and Linear Bottlenecks) : https://arxiv.org/abs/1801.04381v4
#(EfficientNet) : https://arxiv.org/pdf/1905.11946v5.pdf

'''
@classmethod
# https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
def glorot_uniform(cls, *shape, **kwargs): return cls((np.random.default_rng().random(size=shape, dtype=np.float32) * 2 - 1) * ((6/(shape[0]+prod(shape[1:])))**0.5), **kwargs)
'''


class MBConvBlock :
  def __init__(self,kernel_size,strides,expand_ratio,input_filters,output_filters,se_ratio,has_se,track_running_stats = True):
    if expand_ratio != 1 :
      oup = expand_ratio*input_filters #number of output channels
      self._expand_ratio = nn.init.xavier_uniform_(oup,input_filters,1,1)
      self._bn0 = nn.BatchNorm2d(oup,track_running_stats = track_running_stats)
    else :
      self._expand_conv = None

    self.strides = strides
    if strides == (2,2):
      self.pad = [(kernel_size-1)//2-1, (kernel_size-1)//2]*2
    else :
      self.pad =  [(kernel_size-1)//2]*4

    self._depthwise_conv = nn.init.xavier_uniform_(oup,1,kernel_size,kernel_size)
    self._bn1 = nn.BatchNorm2d(oup,track_running_stats = track_running_stats)

    self.has_se = has_se

    if self.has_se :
      num_squeezed_channels = max(1, int(input_filters * se_ratio))
      self._se_reduce = nn.init.xavier_uniform_(num_squeezed_channels, oup, 1, 1)
      self._se_reduce_bias = torch.zeros(num_squeezed_channels)
      self._se_expand = nn.init.xavier_uniform_(oup, num_squeezed_channels, 1, 1)
      self._se_expand_bias = torch.zeros(oup)


    self._project_conv = nn.init.xavier_uniform_(output_filters, oup, 1, 1)
    self._bn2 = nn.BatchNorm2d(output_filters, track_running_stats=track_running_stats)


  swish = nn.SiLU()

  def __call__(self,inputs):
    x = inputs
    if self._expand_conv :
      x = swish(self._bn0(x.Conv2d(self._expand_conv)))

    x = x.Conv2d(self._depthwise_conv, padding=self.pad, stride=self.strides, groups=self._depthwise_conv.shape[0])
    x = swish(self._bn1(x))

    if self.has_se:
      x_squeezed = nn.Avg_Pool2d(kernel_size=x.shape[2:4])
      x_squeezed = x_squeezed.Conv2d(self._se_reduce, self._se_reduce_bias).swish()
      x_squeezed = x_squeezed.Conv2d(self._se_expand, self._se_expand_bias)
      x = torch.mul(x,torch.sigmoid(x_squeezed))

    x = self._bn2(x.Conv2d(self._project_conv))

    if x.shape == inputs.shape:
      x = torch.add(x,inputs)
    return x




class EfficientNet:
  def __init__(self,number=0,classes = 1000,has_se = True,track_running_stats=True,input_channels = 3 , has_fc_output = True):
    self.number=number
    global_params = [
      # width, depth
      (1.0, 1.0), # b0
      (1.0, 1.1), # b1
      (1.1, 1.2), # b2
      (1.2, 1.4), # b3
      (1.4, 1.8), # b4
      (1.6, 2.2), # b5
      (1.8, 2.6), # b6
      (2.0, 3.1), # b7
      (2.2, 3.6), # b8
      (4.3, 5.3), # l2
    ][max(number,0)]

    def round_filters(filters):
      multiplier = global_params[0]
      divisor = 8
      filters *= multiplier
      new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
      if new_filters < 0.9 * filters: # prevent rounding by more than 10%
        new_filters += divisor
      return int(new_filters)

    def round_repeats(repeats):
      return int(math.ceil(global_params[1] * repeats))

    out_channels = round_filters(32)


    blocks_args = [
      [1, 3, (1,1), 1, 32, 16, 0.25],
      [2, 3, (2,2), 6, 16, 24, 0.25],
      [2, 5, (2,2), 6, 24, 40, 0.25],
      [3, 3, (2,2), 6, 40, 80, 0.25],
      [3, 5, (1,1), 6, 80, 112, 0.25],
      [4, 5, (2,2), 6, 112, 192, 0.25],
      [1, 3, (1,1), 6, 192, 320, 0.25],
    ]

    if self.number == -1:
      blocks_args = [
        [1, 3, (2,2), 1, 32, 40, 0.25],
        [1, 3, (2,2), 1, 40, 80, 0.25],
        [1, 3, (2,2), 1, 80, 192, 0.25],
        [1, 3, (2,2), 1, 192, 320, 0.25],
      ]
    elif self.number == -2:
      blocks_args = [
        [1, 9, (8,8), 1, 32, 320, 0.25],
      ]

    self._block = []
    for num_repeats, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio in blocks_args:
      input_filters, output_filters = round_filters(input_filters), round_filters(output_filters)
      for n in range(round_repeats(num_repeats)):
        self._blocks.append(MBConvBlock(kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio, has_se=has_se, track_running_stats=track_running_stats))
        input_filters = output_filters
        strides = (1,1)

    in_channels = round_filters(320)
    out_channels = round_filters(1280)

    def load_from_pretrained(self):
      model_urls = {
        0: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth",
        1: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth",
        2: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth",
        3: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth",
        4: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth",
        5: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth",
        6: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth",
        7: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth"
      }


