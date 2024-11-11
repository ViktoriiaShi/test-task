#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, feature
from skimage.transform import integral_image
from skimage.feature import match_descriptors, ORB
from sklearn.neighbors import NearestNeighbors


# in this task i am using ORB approach (Oriented FAST and rotated BRIEF). it is commonly used in computer vision tasks such as object recognition and image matching. other types of approaches that could be used are siamese neural network (designed to determine whether two pairs of data have similarities or not) and convolutional neural network (useful for finding patterns in images)

# the dataset from kaggle is 35gb, i couldn't load it on my computer because i have got not enough space for it now

# In[ ]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("isaienkov/deforestation-in-ukraine")

print("Path to dataset files:", path)


# In[3]:


# the function loads an image from file and converts it to grayscale
def load_image(image_path):
    image = io.imread(image_path)
    gray_image = color.rgb2gray(image)
    return image, gray_image


# In[4]:


# the function detects keypoints and computes descriptors using ORB (handling image rotations)
def detect_and_compute(image_gray):
    orb = ORB(n_keypoints=500, fast_threshold=0.05)  # Adjust number of keypoints and threshold
    orb.detect_and_extract(image_gray)
    keypoints = orb.keypoints
    descriptors = orb.descriptors
    return keypoints, descriptors


# In[5]:


# match descriptors using NearestNeighbors (alternative to FLANN)
def match_descriptors_scikit(des1, des2):
    # Using Nearest Neighbors for descriptor matching
    nn = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean')
    nn.fit(des2)
    distances, indices = nn.kneighbors(des1)
    return distances, indices


# In[6]:


# filter matches using Lowe's ratio test
def filter_matches(distances, indices, ratio_threshold=0.75):
    good_matches = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if dist[0] < ratio_threshold * dist[1]:  # Lowe's ratio test
            good_matches.append(i)
    return good_matches, indices[good_matches]


# In[7]:


# function for drawing the matched keypoints between the two images
def draw_matches(img1, kp1, img2, kp2, good_matches, indices):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot both images side by side
    ax.imshow(np.concatenate((img1, img2), axis=1))
    
    # Plot the keypoints
    for i in good_matches:
        # kp1 and kp2 are the keypoints from each image
        ax.plot(kp1[i, 1], kp1[i, 0], 'ro')  # Red dots for keypoints in image 1
        ax.plot(kp2[indices[i], 1] + img1.shape[1], kp2[indices[i], 0], 'bo')  # Blue dots for keypoints in image 2
        ax.plot([kp1[i, 1], kp2[indices[i], 1] + img1.shape[1]], 
                [kp1[i, 0], kp2[indices[i], 0]], 'g-', lw=1)  # Green lines for matches
    
    ax.set_title('Image Matching with ORB')
    ax.axis('off')
    plt.show()


# In[ ]:


def main(image_path1, image_path2):
    # Step 1: Load the images
    img1, gray1 = load_image(image_path1)
    img2, gray2 = load_image(image_path2)

    # Step 2: Detect and compute keypoints and descriptors
    kp1, des1 = detect_and_compute(gray1)
    kp2, des2 = detect_and_compute(gray2)

    # Step 3: Match descriptors
    distances, indices = match_descriptors_scikit(des1, des2)

    # Step 4: Filter good matches using Lowe's ratio test
    good_matches, matched_indices = filter_matches(distances, indices)

    # Step 5: Visualize the matches
    draw_matches(img1, kp1, img2, kp2, good_matches, matched_indices)

if __name__ == "__main__":
    image_path1 = "image1.tif"  # Path to the first image
    image_path2 = "image2.tif"  # Path to the second image
    main(image_path1, image_path2)

