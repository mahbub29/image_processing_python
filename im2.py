
import cv2
import numpy as np


def mouse_callback(event, i, j, flags, params):
    # left click event value is 1
    # right click event value is 2
    if event == 1:
        global left_clicks
        left_clicks.append([i, j])
        print(left_clicks)


def kmeans_grayscale_segmentation(image, intensity_vector):
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


def kmeans_image_segmentation(image, init_intensity_vector, ndims=3):
    # set arbitrary previous intensity vector to initially compare to
    prev_intensity_vector = np.ones([init_intensity_vector.size,1])*1000
    intensity_vector = init_intensity_vector

    if len(image.shape)<3:
        while np.prod(intensity_vector) != np.prod(prev_intensity_vector):
            prev_intensity_vector = intensity_vector
            output, intensity_vector2 = kmeans_grayscale_segmentation(image, intensity_vector)
            print(intensity_vector)
    else:
        # continue to loop whilst previous and new intensity vector products are not equal
        while np.prod(intensity_vector) != np.prod(prev_intensity_vector):
            prev_intensity_vector = intensity_vector
            output, intensity_vector = kmeans_nD_segmentation(image, intensity_vector, ndims=ndims)
            print(intensity_vector)

    return output, intensity_vector


def kmeans_nD_segmentation(image, i_vec, ndims=3):
    height, width, rgb = image.shape
    image_thread = image.reshape([1, height*width, 3])

    # check if request 5D k-means (RGBij)
    if ndims==5:
        image_thread_ij = np.zeros([1,height*width,2])
        count = 0
        for i in range(height):
            for j in range(width):
                image_thread_ij[0,count,:] = np.array([[i,j]])
                count += 1
        image_thread = np.dstack((image_thread, image_thread_ij))
    else:
        pass

    # make dictionaries to store the cluster intensity sums as well as the
    # total number belonging to each cluster
    k_sums, k_tots = dict(), dict()

    # calculate the root square distance (in terms of RGBij) each pixel in the image
    # to each of the selected pixel RGB values
    # i.e. d = sqrt(r^2 + g^2 + b^2 + i^2 + j^2)
    # assign a k cluster number to each set of RGB differences
    delta = np.zeros([int(i_vec.shape[0]), height*width])
    label_array = np.ones(delta.shape)
    for i in range(int(i_vec.shape[0])):
        if ndims==5:
            k_sums[i+1] = np.array([[0,0,0,0,0]]) # 1 by 5 array summation for RGBij values (BGR for python)
        else:
            k_sums[i+1] = np.array([[0,0,0]])     # 1 by 3 array summation for RGB values (BGR for python)
        k_tots[i+1] = 0
        delta[i,:] = np.sqrt(np.sum((image_thread - np.array(i_vec[i,:])).astype(int)**2, axis=2))
        label_array[i,:] = label_array[i,:]*(i+1)
    pixel_labels_3darray = np.dstack((delta, label_array))

    # initialise an array to contain the final pixel cluster identities
    pixel_labels_final = np.zeros([1, height*width])

    # find the lowest delta_rgbij out of the k groups and label the pixel as
    # belonging to the k value with the lowest delta rgb value
    for i in range(height*width):
        p = np.where(pixel_labels_3darray[:,i,0]==np.amin(pixel_labels_3darray[:,i,0], axis=0))
        pixel_labels_final[0,i] = pixel_labels_3darray[int(p[0][0]),i,1]
        # increment the cluster group dictionary value by one
        k_tots[int(pixel_labels_final[0,i])] += 1
        # add the actual pixel intensity to the cluster dictionary item
        k_sums[int(pixel_labels_final[0,i])] += image_thread[0,i,:].astype(int)

    # calculate the new average pixel intensity vector and return it
    for i in range(i_vec.shape[0]):
        for c in range(i_vec.shape[1]):
            try:
                i_vec[i,c] = np.around(k_sums[i+1][0,c]/k_tots[i+1])
            except ZeroDivisionError:
                i_vec[i,c] = int(0)
            except ValueError:
                i_vec[i, c] = int(0)

    # get the output image
    output = np.zeros([1,height*width,3])
    for i in range(height*width):
        output[0,i,:] = i_vec[int(pixel_labels_final[0,i]-1),:3]

    output = output/np.max(output, axis=1)
    output = output.reshape([height,width,rgb])

    return output, i_vec


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
                        init_intensities_5d = list()

                        for i, j in left_clicks:
                            init_intensities.append(self.image[j, i])
                            if len(self.image.shape)==3:
                                init_intensities_5d.append(np.ndarray.tolist(self.image[j, i]))
                            else:
                                pass

                        # for 5D include the pixel coordinates to the initial selected intensitites
                        # i.e. the intensity dimensions will be [B, G, R, i, j]
                        if len(self.image.shape)==3:
                            for i in range(len(left_clicks)):
                                init_intensities_5d[i].append(left_clicks[i][1])
                                init_intensities_5d[i].append(left_clicks[i][0])
                            init_intensities_5d = np.array(init_intensities_5d)

                        if init_intensities[0].size < 3:
                            ### FOR GRAYSCALE IMAGE ###
                            init_intensities = np.array(init_intensities).reshape([len(init_intensities), 1])
                            print(init_intensities)
                            out, i_vec = kmeans_image_segmentation(self.image, init_intensities)
                            print(i_vec)
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

                            out3d, i_vec3d = kmeans_image_segmentation(self.image, init_intensities)
                            out5d, i_vec5d = kmeans_nD_segmentation(self.image, init_intensities_5d, ndims=5)
                            print('FINAL 3D VECTOR')
                            print(i_vec3d)
                            print('FINAL 5D VECTOR')
                            print(i_vec5d)
                            print("FINISHED")
                            cv2.imshow('3D output - %s colors' % str(i_vec3d.shape[0]), out3d)
                            cv2.imshow('5D output - %s colors' % str(i_vec5d.shape[0]), out5d)

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
    img = cv2.imread('sample_image_t1.jpg',0)
    print(img.shape)

    out = SegmentedImage(img)
    out.kmeams_segmentation(3)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()