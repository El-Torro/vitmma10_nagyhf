import cv2
import numpy as np

# Compute scaled absolute image gradients
def create_sobels(image, sobel_kernel=3):
    # Convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute x and y derivatives using sobel
    sobelx = cv2.Sobel (gray, cv2.CV_32F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, sobel_kernel)

    # Get absolute value
    sobelx_abs = np.absolute(sobelx)
    sobely_abs = np.absolute(sobely)

    # Compute scaling factor
    scale_x = 255 / np.max(sobelx_abs)
    scale_y = 255 / np.max(sobely_abs)

    # Scale gradients
    sobelx_scaled = scale_x*sobelx_abs
    sobely_scaled = scale_y*sobely_abs

    return sobelx_abs,sobely_abs,sobelx_scaled,sobely_scaled
    
# Threshold gradients based on magintude
def mag_threshold(scaled_sobelx, scaled_sobely, mag_thresh=(30, 100)):
    # Compute gradient magnitude
    mag = scaled_sobelx + scaled_sobely

    # Threshold using inRange
    mag_binary = cv2.inRange(mag, mag_thresh[0] * 2, mag_thresh[1] * 2)

    return mag_binary
    
# Threshold gradients based on direction
def dir_threshold(abs_sobelx, abs_sobely, thresh=(0, np.pi/3)):
    # Compute gradient direction
    dir = np.arctan2 (abs_sobely, abs_sobelx)

    # Threshold using inRange
    dir_binary = cv2.inRange(dir, thresh[0], thresh[1])

    return dir_binary
    
# Apply color threshold
def color_threshold(image):
    thresh=(200,255)
    # Convert to hls
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # Get saturation
    image_s = image_hls[:, :, 2]

    # Threshold using inRange
    color_binary = cv2.inRange(image_s, thresh[0], thresh[1])

    return color_binary
    
# Combine all kinds of thresholds
def apply_thresholds(image, ksize=3):
    # Get derivatives
    abs_sobelx, abs_sobely, scaled_sobelx, scaled_sobely = create_sobels(image)
    
    # Compute magnitude and direction threshold
    mag_thresh = mag_threshold(scaled_sobelx, scaled_sobely)
    dir_thresh = dir_threshold(abs_sobelx, abs_sobely)
    
    # Compute color threshold
    color_thresh = color_threshold(image)

    # Combine all thresholded images
    combined = np.zeros_like(abs_sobelx)
    combined[((mag_thresh == 255) & (dir_thresh == 255)) | (color_thresh == 255)] = 1

    return combined

def warp(img):
    img_size = (img.shape[1], img.shape[0])

    # Source and destination points
    src = np.float32(
        [[380, 0],
          [875, 235],
          [60, 235],
          [470, 0]])

    dst = np.float32(
        [[150, 0],
         [800, 260],
         [150, 260],
         [800, 0]])

    # Get perspective transforms
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp image
    binary_warped = cv2.warpPerspective(img, M, img_size)#, flags=cv2.INTER_LINEAR)
    
    return binary_warped, Minv
    

def get_histogram(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)

    return histogram
 
