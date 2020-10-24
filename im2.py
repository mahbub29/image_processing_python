
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
        output, intensity_vector = kmeans_segmentation_process(image, intensity_vector)

    return output, intensity_vector


def color_image_segmentation(image, colors):
    image_thread = image.reshape([1,image[:,:,0].size,3])
    for i in range(image_thread[:,:,0].size):
        diffs = np.abs(np.sum(colors, axis=1) - np.sum(image_thread[0,i,:]))
        cluster_identity = int(np.where(diffs==np.amin(diffs, axis=0))[0][0])
        image_thread[0,i,:] = colors[cluster_identity,:]
    output = image_thread.reshape(image.shape)
    print('out')
    print(output[:,:,0])
    return output


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
                            out, i_vector = kmeans_image_segmentation(self.image, init_intensities)
                            cv2.imshow('output', out)
                        else:
                            ### FOR RGB IMAGE ###
                            # separate the intensities into their RGB components
                            for i in range(len(init_intensities)):
                                if i==0:
                                    intensity_vector = init_intensities[0]
                                else:
                                    intensity_vector = np.concatenate((intensity_vector,init_intensities[i]))
                            init_intensities = intensity_vector.reshape([len(init_intensities),3])
                            init_blue = np.array(init_intensities[:,0]).reshape([len(init_intensities[:,0]),1])
                            init_green = np.array(init_intensities[:,1]).reshape([len(init_intensities[:,1]),1])
                            init_red = np.array(init_intensities[:,2]).reshape([len(init_intensities[:,2]),1])

                            # output the RGB components
                            blue_out, blue_vec = kmeans_image_segmentation(self.image[:,:,0], init_blue)
                            green_out, green_vec = kmeans_image_segmentation(self.image[:,:,1], init_green)
                            red_out, red_vec = kmeans_image_segmentation(self.image[:,:,2], init_red)
                            colors = np.concatenate((blue_vec, green_vec, red_vec), axis=1)
                            print(colors)

                            # combine RGB components for the final image
                            # This uses only k distinct colors from the variable colors based
                            # on how many pixels have been selected
                            out1 = color_image_segmentation(self.image, colors)
                            # This uses k different values for each RGB, but combines them to make
                            # different colors based on how close the RGB values are to the calculated RGBs
                            out2 = np.dstack((blue_out, green_out, red_out))
                            np.savetxt('out2blue.csv', np.around(out2[:,:,0]*255), delimiter=',')
                            np.savetxt('out2green.csv', np.around(out2[:,:,1]*255), delimiter=',')
                            np.savetxt('out2red.csv', np.around(out2[:,:,2]*255), delimiter=',')
                            print(out2[:,:,0])
                            cv2.imshow('output - k colors', out1)
                            cv2.imshow('output - k RGB values', out2)

                        print("FINISHED")
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