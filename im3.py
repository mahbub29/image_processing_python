import numpy as np
import cv2
import matplotlib.pyplot as plt


def histogram_match(source, target):
    cdfSource = HistogramProcesses(source).get_cumulative_histogram()
    cdfTarget = HistogramProcesses(target).get_cumulative_histogram()



class HistogramProcesses:
    def __init__(self, image):
        self.image = image
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

    def get_histogram(self):
        I = self.image.reshape([self.image.size, 1])
        intensities = dict()
        for i in range(256):
            intensities[i] = 0
        for i in range(I.size):
            intensities[int(I[i])] += 1
        normalized_intensities = dict()
        for key, val in intensities.items():
            normalized_intensities[key] = val/self.image.size
        return intensities, normalized_intensities

    def get_cumulative_histogram(self):
        intensities, normalized_intensities = self.get_histogram()
        cumulative_intensities = dict()
        for key, val in intensities.items():
            if key == 0:
                pass
            else:
                intensities[key] += intensities[key-1]
            cumulative_intensities[key] = intensities[key]/self.image.size
        return cumulative_intensities

    def get_equalized_image(self):
        cdf = self.get_cumulative_histogram()
        copy = self.image.reshape([self.image.size, 1])
        for v in range(copy.size):
            copy[v] = cdf[int(copy[v])]*255
        copy = (copy/np.max(copy)).reshape(self.image.shape)
        return copy


if __name__=="__main__":
    file = 'C:\\Users\\mahbu\\OneDrive\\Pictures\\Camera Roll\\WIN_20200917_14_03_54_Pro.jpg'
    img = cv2.imread(file, 0)
    cv2.imshow('in', img)
    I = HistogramProcesses(img)
    h, nh = I.get_histogram()
    out = I.get_equalized_image()
    cv2.imshow('out', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # out = np.around(out*255)
    # O = HistogramProcesses(out)
    # oh, onh = O.get_histogram()
    # plt.plot(onh.keys(), onh.values(), nh.keys(), nh.values())
    # plt.show()
