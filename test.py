import math
Hin = 1
padding = 0
kernel_size = 1
stride = 1
dilation = 1
Hout = math.floor((Hin + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
print(Hout)