import numpy as np
import struct
import matplotlib.pyplot as plt

filename = 'train-images-idx3-ubyte'
binfile = open(filename, 'rb')
buf = binfile.read()
# print(buf)

index = 0
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
index += struct.calcsize('>IIII')

for i in range(10):
    im = struct.unpack_from('>784B', buf, index)
    index += struct.calcsize('>784B')

    im = np.array(im)
    im = im.reshape(28, 28)
    # print(im)

    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.imshow(im, cmap='gray')
    plt.show()

# fig, ax = plt.subplots(
#     nrows=2,
#     ncols=5,
#     sharex=True,
#     sharey=True, )
#
# ax = ax.flatten()
# for i in range(10):
#     img = X_train[y_train == i][0].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()