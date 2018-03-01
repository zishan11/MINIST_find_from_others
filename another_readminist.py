import numpy as np
import struct


def loadImageSet(filename):
    print("load image set", filename)
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)
    print("head,", head)

    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    # [60000]*28*28
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'  # like '>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, 1, width * height])
    print("load imgs finished")
    return imgs


def loadLabelSet(filename):
    print("load label set", filename)
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)
    print("head,", head)
    imgNum = head[1]

    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])

    print('load label finished')
    return labels


if __name__ == "__main__":
    imgs = loadImageSet("train-images-idx3-ubyte")
    labels = loadLabelSet("train-labels-idx1-ubyte")

    # imgs = loadImageSet("t10k-images.idx3-ubyte")
    # labels = loadLabelSet("t10k-labels.idx1-ubyte")

# 基于python2
# [python]
# view
# plain
# copy
# import numpy as np
# import struct
#
#
# def loadImageSet(filename):
#     print
#     "load image set", filename
#     binfile = open(filename, 'rb')
#     buffers = binfile.read()
#
#     head = struct.unpack_from('>IIII', buffers, 0)
#     print
#     "head,", head
#
#     offset = struct.calcsize('>IIII')
#     imgNum = head[1]
#     width = head[2]
#     height = head[3]
#     # [60000]*28*28
#     bits = imgNum * width * height
#     bitsString = '>' + str(bits) + 'B'  # like '>47040000B'
#
#     imgs = struct.unpack_from(bitsString, buffers, offset)
#
#     binfile.close()
#     imgs = np.reshape(imgs, [imgNum, 1, width * height])
#     print
#     "load imgs finished"
#     return imgs
#
#
# def loadLabelSet(filename):
#     print
#     "load label set", filename
#     binfile = open(filename, 'rb')
#     buffers = binfile.read()
#
#     head = struct.unpack_from('>II', buffers, 0)
#     print
#     "head,", head
#     imgNum = head[1]
#
#     offset = struct.calcsize('>II')
#     numString = '>' + str(imgNum) + "B"
#     labels = struct.unpack_from(numString, buffers, offset)
#     binfile.close()
#     labels = np.reshape(labels, [imgNum, 1])
#
#     print
#     'load label finished'
#     return labels
#
#
# if __name__ == "__main__":
#     imgs = loadImageSet("train-images.idx3-ubyte")
#     labels = loadLabelSet("train-labels.idx1-ubyte")

# # 为了测试是否读取成功只读取一张图片的python
# # 代码
# # [python]
# # view
# # plain
# # copy
# import numpy as np
# import struct
# import matplotlib.pyplot as plt
#
# filename = 'train-images.idx3-ubyte'
# binfile = open(filename, 'rb')
# buf = binfile.read()
#
# index = 0
# magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
# index += struct.calcsize('>IIII')
#
# im = struct.unpack_from('>784B', buf, index)
# index += struct.calcsize('>784B')
#
# im = np.array(im)
# im = im.reshape(28, 28)
#
# fig = plt.figure()
# plotwindow = fig.add_subplot(111)
# plt.imshow(im, cmap='gray')
# plt.show()


