from matplotlib import pyplot as plt
import time

class Plot(object):
    count = 1

    def __init__(self, title, xlabel, ylabel):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        plt.figure(Plot.count)
        self.count = Plot.count
        Plot.count = Plot.count + 1
        plt.plot([], [])
        plt.ion() # needed, otherwise .show() will freeze the execution
        plt.show()

    def update(self, x, y, labels):
        plt.figure(self.count)
        plt.gca().cla()
        for (x0, y0, l0) in zip(x, y, labels):
            plt.plot(x0, y0, label=l0)
        plt.legend(loc='best')
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.draw()

    def save(self, out):
        plt.figure(self.count).savefig(out + self.title + '-' + str(int(time.time())) + '.png')
