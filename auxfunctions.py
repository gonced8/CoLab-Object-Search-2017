import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
from matplotlib import colors


def probGraph(prediction):
    # make values from 0 to 1, for this example
    vals = prediction.reshape(16, 16)

    # make a color map of colors
    cmap = colors.LinearSegmentedColormap.from_list('my_colormap',
                                                    ['black', 'white'],
                                                    256)

    # tell imshow about color map so that only set colors are used
    img = pyplot.imshow(vals, interpolation='nearest',
                        cmap=cmap,
                        origin='lower')

    pyplot.colorbar(img, cmap=cmap)

    pyplot.show()