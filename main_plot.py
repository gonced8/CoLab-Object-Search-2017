import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot
from matplotlib import colors
from matplotlib.colors import Normalize
import numpy as np


def plot_image():
    image = np.load('out/outfile0.npz')['x_train'][0]
    norm = Normalize()
    norm.autoscale(image)
    img_norm = norm(image)

    pyplot.imshow(img_norm, origin='upper', extent=[0, 16, 0, 16])


def plot_sequence():
    b = np.load('sequence/seq0.npz')
    seq = b['x_sequence_index'][0][9]
    # seq = [1, 2, 255]
    matrix = [[0 for n in range(16)] for m in range(16)]
    for i in range(len(seq)):
        x = int(seq[i] % 16)
        y = int(seq[i] / 16)
        matrix[y][x] = i + 1

    # make a color map of colors
    cmap = colors.LinearSegmentedColormap.from_list('my_colormap',
                                                ['white', 'red'],
                                                256)


    # tell imshow about color map so that only set colors are used
    img = pyplot.imshow(matrix, interpolation='nearest',
                        cmap=cmap,
                        origin='upper',
                        alpha=0.5,
                        extent=[0, 16, 0, 16])

    pyplot.colorbar(img, cmap=cmap)


def plot_sequence_on_image():
    plot_image()
    plot_sequence()
    pyplot.show()


plot_sequence_on_image()


# transformar posicao da sequencia em posicao na imagem
# atribuir valor de 0 a max a cada uma das posicoes por ordem
# criar grid de tamanho 256 com tudo a zeros
# fazer plot com escala de cores de a max
