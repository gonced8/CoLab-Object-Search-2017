import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot
from matplotlib import colors
import matplotlib.image as mpimg
from keras.preprocessing import image
import numpy as np


def plotImage():
    a = np.load('out/outfile0.npz')['x_train'][0]
    x = image.array_to_img(a)
    arrayForReal = image.img_to_array(a)

    img = mpimg.imread("test.png")

    pyplot.imshow(img, extent=[0,16, 0,16])


seq = [20, 45, 100]
matrix = [[0 for n in range(16)] for m in range(16)]
for i in range(len(seq)):
    x = seq[i] % 16
    y = seq[i] // 16
    print(x)
    print(y)
    print(i)
    matrix[x][y] = i + 1

# make a color map of colors
cmap = colors.LinearSegmentedColormap.from_list('my_colormap',
                                                ['white', 'red'],
                                                256)


plotImage()

# tell imshow about color map so that only set colors are used
img = pyplot.imshow(matrix, interpolation='nearest',
                    cmap=cmap,
                    origin='lower',
                    alpha=0.5,
                    extent=[0, 16, 0, 16])

pyplot.colorbar(img, cmap=cmap)



pyplot.show()

# x.show()
# img.show(x)


# transformar posicao da sequencia em posicao na imagem
# atribuir valor de 0 a max a cada uma das posicoes por ordem
# criar grid de tamanho 256 com tudo a zeros
# fazer plot com escala de cores de a max
