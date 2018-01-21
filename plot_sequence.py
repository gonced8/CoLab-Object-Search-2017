import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot
from matplotlib import colors
from matplotlib.colors import Normalize
import numpy as np


def plot_image(image_number):
    file_name = 'out/outfile' + str(image_number) + '.npz'
    image = np.load(file_name)['x_train'][0]
    norm = Normalize()
    norm.autoscale(image)
    img_norm = norm(image)

    pyplot.imshow(img_norm, origin='upper', extent=[0, 16, 0, 16])


def plot_sequence(seq_number):
    file_name = 'sequence/seq' + str(seq_number) + '.npz'
    aux = np.load(file_name)
    seq = aux['x_sequence_index'][0][9]

    matrix = np.zeros((16, 16))
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


def plot_sequence_on_image(image_number):
    plot_image(image_number)
    plot_sequence(image_number)
    pyplot.show()


if __name__ == '__main__':
    plot_sequence_on_image(7)
