import cv2
import numpy as np


def get_window(image, window_size, i, j):
    height = image.shape[0]
    width = image.shape[1]
    limit = window_size//2

    if i<limit:
        if j<limit:
            window = image[0:i+limit+1, 0:j+limit+1]
        elif j>width-limit-1:
            window = image[0:i+limit+1, j-limit:width]
        else:
            window = image[0:i+limit+1, j-limit:j+limit+1]
    elif j<limit:
        if i<limit:
            window = image[0:i+limit+1, 0:j+limit+1]
        elif i>height-limit:
            window = image[i-limit:height, 0:j+limit+1]
        else:
            window = image[i-limit:i+limit+1, 0:j+limit+1]
    elif i>height-limit-1:
        if j<limit:
            window = image[i-limit:height, 0:j+limit+1]
        elif j>width-limit-1:
            window = image[i-limit:height, j-limit:width]
        else:
            window = image[i-limit:height, j-limit:j+limit+1]
    elif j>width-limit-1:
        if i<limit:
            window = image[0:i+limit+1, j-limit:width]
        elif i>height-limit-1:
            window = image[i-limit:height, j-limit:width]
        else:
            window = image[i-limit:i+limit+1, j-limit:width]
    else:
        window = image[i-limit:i+limit+1, j-limit:j+limit+1]
    return window


def get_median(window):
    window = window.reshape(1, window.size)
    if window.size%2==1:
        median = sorted(window)[0][window.size//2]
    else:
        median = (int(sorted(window)[0][window.size//2]) +
                  int(sorted(window)[0][window.size//2+1]))/2
    return median


class ProcessImage:
    def __init__(self, image, window_size):
        self.image = image
        self.window_size = window_size
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

    def median_filter(self, image, window_size):
        output_image = np.zeros(image.shape)
        for i in range(self.height):
            for j in range(self.width):
                window = get_window(image, window_size, i, j)
                output_image[i, j] = get_median(window)
        output_image = output_image / np.max(output_image)
        return output_image

    def gray_median_filter(self):
        print("Processing GRAY Median Filter")
        return self.median_filter(self.image, self.window_size)

    def rgb_median_filter(self):
        print("Processing RGB Median Filter")
        output_image = np.zeros(self.image.shape)
        # blue channel
        output_image[:, :, 0] = self.median_filter(self.image[:, :, 0], self.window_size)
        # green channel
        output_image[:, :, 1] = self.median_filter(self.image[:, :, 1], self.window_size)
        # red channel
        output_image[:, :, 2] = self.median_filter(self.image[:, :, 2], self.window_size)
        return output_image

    def means_and_variances(self, image, window_size):
        means, variances = np.zeros(image.shape), np.zeros(image.shape)
        for i in range(self.height):
            for j in range(self.width):
                window = get_window(image, window_size, i, j)
                means[i,j] = np.sum(window)/window.size
                variances[i,j] = np.sum(window**2 - means[i,j]**2)/window.size
        return means, variances

    def adaptive_filter(self, image, window_size):
        mean, variance = self.means_and_variances(image, window_size)
        output_image = np.zeros(image.shape)
        for i in range(self.height):
            for j in range(self.width):
                variance_window = get_window(variance, window_size, i, j)
                output_image[i,j] = mean[i,j]\
                                    + ((variance[i,j]-(np.sum(variance_window)/variance_window.size))/variance[i,j])\
                                    * (image[i,j]-mean[i,j])
        output_image = output_image / np.max(output_image)
        return output_image

    def gray_adaptive_filter(self):
        print("Processing GRAY Adaptive Filter")
        return self.adaptive_filter(self.image, self.window_size)

    def rgb_adaptive_filter(self):
        print("Processing RGB Adaptive Filter")
        output_image = np.zeros(self.image.shape)
        # blue channel
        output_image[:, :, 0] = self.adaptive_filter(self.image[:, :, 0], self.window_size)
        # green channel
        output_image[:, :, 1] = self.adaptive_filter(self.image[:, :, 1], self.window_size)
        # red channel
        output_image[:, :, 2] = self.adaptive_filter(self.image[:, :, 2], self.window_size)
        return output_image


if __name__=="__main__":
    img = cv2.imread('sample_image_t1.jpg')
    window_size = 15
    out = ProcessImage(img, window_size)

    cv2.imshow('image_in', img)
    cv2.imshow('image_out', out.gray_adaptive_filter())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
