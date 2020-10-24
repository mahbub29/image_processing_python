import numpy as np
import  cv2
from im1 import get_window
from im3 import HistogramProcesses

class ImageKernel:
    def __init__(self, image):
        self.image = image
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        window_size = 3

        self.sharpen = np.zeros(image.shape)
        self.blur = np.zeros(image.shape)
        self.emboss = np.zeros(image.shape)
        self.top_sobel = np.zeros(image.shape)
        self.bottom_sobel = np.zeros(image.shape)
        self.left_sobel = np.zeros(image.shape)
        self.right_sobel = np.zeros(image.shape)
        self.outline = np.zeros(image.shape)
        self.smooth = np.zeros(image.shape)

        sharpening_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        blur_kernel = np.array([[0.0625, 0.125, 0.0625],[0.125, 0.25, 0.125],[0.0625, 0.125, 0.0625]])
        emboss_kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
        top_sobel_kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        bottom_sobel_kernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        left_sobel_kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        right_sobel_kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        outline_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        smoothing_kernel = np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])

        for i in range(1, self.height-1):
            for j in range(1, self.width-1):
                window = get_window(self.image, window_size, i,j)
                self.sharpen[i,j] = np.sum(window*sharpening_kernel)
                self.blur[i,j] = np.sum(window*blur_kernel)
                self.emboss[i,j] = np.sum(window*emboss_kernel)
                self.top_sobel[i,j] = np.sum(window*top_sobel_kernel)
                self.bottom_sobel[i,j] = np.sum(window*bottom_sobel_kernel)
                self.left_sobel[i,j] = np.sum(window*left_sobel_kernel)
                self.right_sobel[i,j] = np.sum(window*right_sobel_kernel)
                self.outline[i,j] = np.sum(window*outline_kernel)
                self.smooth[i,j] = np.sum(window*smoothing_kernel)

        self.sharpen = self.sharpen/np.max(self.sharpen)
        self.blur = self.blur/np.max(self.blur)
        self.emboss = self.emboss/np.max(self.emboss)
        self.top_sobel = self.top_sobel/np.max(self.top_sobel)
        self.bottom_sobel = self.bottom_sobel/np.max(self.bottom_sobel)
        self.left_sobel = self.left_sobel/np.max(self.left_sobel)
        self.right_sobel = self.right_sobel/np.max(self.right_sobel)
        self.outline = self.outline/np.max(self.outline)
        self.smooth = self.smooth/np.max(self.smooth)


if __name__=="__main__":
    img = cv2.imread('moon.jpg', 0)
    # img = HistogramProcesses(img).get_equalized_image()
    cv2.imshow('input', img)
    img = ImageKernel(img)
    cv2.imshow('sharpen', img.sharpen)
    # cv2.imshow('blur', img.blur)
    # cv2.imshow('emboss', img.emboss)
    # cv2.imshow('top sobel', img.top_sobel)
    # cv2.imshow('bottom sobel', img.bottom_sobel)
    # cv2.imshow('left sobel', img.left_sobel)
    # cv2.imshow('right sobel', img.right_sobel)
    # cv2.imshow('outline', img.outline)
    # cv2.imshow('smoothed', img.smooth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()