
import cv2
import numpy as np


def mouse_callback(event, i, j, flags, params):
    # left click event value is 1
    # right click event value is 2
    if event == 1:
        global left_clicks
        left_clicks.append([i, j])
        print(left_clicks)


def kmeans_segmentation_process(image, intensity_vector):
    height, width = image.shape
    image_thread = image.reshape([1,image.size])

    # make dictionaries to store the cluster intensity sums as well as the
    # total number belonging to each cluster
    k_sums, k_tots = dict(), dict()

    # set initial k sums and frequencies to 0 and make a overlay k number of image
    # threads over threads containing each cluster identity number
    for i in range(intensity_vector.size):
        k_sums[i+1] = 0
        k_tots[i+1] = 0
        if i==0:
            image_thread_array = image_thread
            label_array = np.ones(image_thread.shape)
        else:
            image_thread_array = np.concatenate((image_thread_array, image_thread))
            label_array = np.concatenate((label_array, np.ones(image_thread.shape)*(i+1)))
    pixel_labels_3darray = np.dstack((image_thread_array, label_array))

    # find the difference between each pixel in the image and each of the selected
    # pixel intensities
    pixel_labels_3darray[:,:,0] = np.abs(pixel_labels_3darray[:,:,0] - intensity_vector)

    # initialise an array to contain the final pixel cluster identities
    pixel_labels_final = np.zeros(image_thread.size)
    for i in range(image_thread.size):
        # store the index of the thread image vector and use this to assign the
        # pixel intensity to a cluster group
        p = np.where(pixel_labels_3darray[:,i,0] == np.amin(pixel_labels_3darray[:,i,0], axis=0))
        pixel_labels_final[i] = pixel_labels_3darray[p[0][0],i,1]
        # increment the cluster group dictionary value by one
        k_tots[pixel_labels_final[i]] += 1
        # add the actual pixel intensity to the cluster dictionary item
        k_sums[pixel_labels_final[i]] += image_thread[0,i]

    # calculate the new average pixel intensity vector and return it
    for i in range(len(intensity_vector)):
        try:
            intensity_vector[i] = round(k_sums[i+1]/k_tots[i+1])
        except ZeroDivisionError:
            intensity_vector[i] = 0

    # get the output image
    output = np.zeros(image_thread.size)
    for i in range(image_thread.size):
        output[i] = intensity_vector[int(pixel_labels_final[i]-1)]
    output = output.reshape([height,width])/np.max(output)

    return output, intensity_vector


def kmeans_image_segmentation(image, init_intensity_vector):
    # set arbitrary previous intensity vector to initially compare to
    prev_intensity_vector = np.ones([init_intensity_vector.size,1])*1000
    intensity_vector = init_intensity_vector

    # continue to loop whilst previous and new intensity vector products are not equal
    while np.prod(intensity_vector) != np.prod(prev_intensity_vector):
        prev_intensity_vector = intensity_vector
        output, intensity_vector = kmeans_3D_color_segmentation(image, intensity_vector)
        print(intensity_vector)

    return output, intensity_vector


def kmeans_3D_color_segmentation(image, intensity_vector):
    height, width, rgb = image.shape
    image_thread = image.reshape([1, height*width, 3])

    # make dictionaries to store the cluster intensity sums as well as the
    # total number belonging to each cluster
    k_sums, k_tots = dict(), dict()

    # calculate the euclidean distance (in terms of RGB) each pixel in the image
    # to each of the selected pixel RGB values
    # i.e. d = sqrt(r^2 + g^2 + b^2)
    # assign a k cluster number to each set of RGB differences
    delta_rgb = np.zeros([len(intensity_vector), height*width])
    label_array = np.ones(delta_rgb.shape)
    for i in range(len(intensity_vector)):
        k_sums[i+1] = np.array([[0,0,0]]) # this must be a 1 by 3 array for RGB values (BGR for python)
        k_tots[i+1] = 0
        delta_rgb[i,:] = np.sqrt(np.sum((image_thread - np.array(intensity_vector[i])).astype(int)**2, axis=2))
        label_array[i,:] = label_array[i,:]*(i+1)
    pixel_labels_3darray = np.dstack((delta_rgb, label_array))

    # initialise an array to contain the final pixel cluster identities
    pixel_labels_final = np.zeros([1, height*width])

    # find the lowest delta_rgb out of the k groups and label the pixel as
    # belonging to the k value with the lowest delta rgb value
    for i in range(height*width):
        p = np.where(pixel_labels_3darray[:,i,0]==np.amin(pixel_labels_3darray[:,i,0], axis=0))
        pixel_labels_final[0,i] = pixel_labels_3darray[int(p[0][0]),i,1]
        # increment the cluster group dictionary value by one
        k_tots[int(pixel_labels_final[0,i])] += 1
        # add the actual pixel intensity to the cluster dictionary item
        k_sums[int(pixel_labels_final[0,i])] += image_thread[0,i,:]

    # calculate the new average pixel intensity vector and return it
    for i in range(len(intensity_vector)):
        for c in range(len(intensity_vector[0])):
            try:
                intensity_vector[i,c] = np.around(k_sums[i+1][0,c]/k_tots[i+1])
            except ZeroDivisionError:
                intensity_vector[i,c] = int(0)
            except ValueError:
                intensity_vector[i, c] = int(0)

    # get the output image
    output = np.zeros(image_thread.shape)
    for i in range(height*width):
        output[0,i,:] = intensity_vector[int(pixel_labels_final[0,i]-1)]
    output = output/np.max(output, axis=1)
    output = output.reshape([height,width,rgb])

    return output, intensity_vector


class SegmentedImage:
    def __init__(self, image):
        self.image = image
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

    def thresholding_basic(self, threshold):
        return (self.image > threshold).astype(float)

    def kmeams_segmentation(self, k):
        global left_clicks
        left_clicks = list()
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('image', mouse_callback)

        notQuit = True
        print("Select a minimum of 2 points.\
        \nPress \"R\" to reset and clear your selection.\
        \nPress \"V\" to verify your selection\
        \nPress \"Q\" to quit.")
        while notQuit:
            cv2.imshow("image", self.image)
            key1 = cv2.waitKey(0)

            # once k means points are selected press "v" to verify complete
            if key1 == ord("v"):
                if len(left_clicks) == 1:
                    print("Only 1 point selected.\
                          \nSelect at least one more point, or press \"R\" to clear the list or \"Q\" to quit.")
                else:
                    print("You have selected %s points.\
                    \nIs this correct?\
                    \nPress \"R\" to clear the list and start again or press \"Y\" to confirm your choices."\
                    % str(len(left_clicks)))
                    key2 = cv2.waitKey(0)
                    # Press "Y" to confirm
                    if key2 == ord("y"):
                        print("RUNNING K-MEANS SEGMENTATION")
                        init_intensities = list()
                        for i, j in left_clicks:
                            init_intensities.append(self.image[j, i])
                        if init_intensities[0].size < 3:
                            ### FOR GRAYSCALE IMAGE ###
                            init_intensities = np.array(init_intensities).reshape([len(init_intensities), 1])
                            out, i_vec = kmeans_image_segmentation(self.image, init_intensities)
                        else:
                            ### FOR RGB IMAGE ###
                            # separate the intensities into their RGB components
                            for i in range(len(init_intensities)):
                                if i==0:
                                    intensity_vector = init_intensities[0]
                                else:
                                    intensity_vector = np.concatenate((intensity_vector,init_intensities[i]))
                            init_intensities = intensity_vector.reshape([len(init_intensities),3])
                            print(init_intensities)

                            out, i_vec = kmeans_image_segmentation(self.image, init_intensities)
                            print(i_vec)

                        print("FINISHED")
                        cv2.imshow('output - k colors', out)
                        cv2.waitKey(0)
                        notQuit = False
                        cv2.destroyAllWindows()
                        break
                    # Press "R" to reset the list
                    elif key2 == ord("r"):
                        print("List of points has been cleared.\
                        \nPlease select at least two new points")
                        left_clicks = list()
                    elif key2 == ord("q"):
                        notQuit = False

            # "r "will reset the list of points selected in the image
            elif key1 == ord("r"):
                print("List of points has been cleared.\
                \nPlease select at least two new points")
                left_clicks = list()
                print(left_clicks)
            elif key1 == ord("q"):
                notQuit = False

        print("DONE")
        # cv2.destroyAllWindows()

if __name__=="__main__":
    file = 'C:\\Users\\mahbu\\OneDrive\\Pictures\\Camera Roll\\WIN_20200917_14_03_54_Pro.jpg'
    img = cv2.imread('Lenna.png')
    print(img.shape)

    out = SegmentedImage(img)
    out.kmeams_segmentation(3)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()