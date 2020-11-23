import cv2
import numpy as np
from labor_img import apply_thresholds
from labor_img import warp
from labor_img import get_histogram


class LineDetector(object):

    def draw_lane_lines(self, original_image, warped_image, Minv, draw_info):
        leftx = draw_info['leftx']
        rightx = draw_info['rightx']
        left_fitx = draw_info['left_fitx']
        right_fitx = draw_info['right_fitx']
        ploty = draw_info['ploty']

        warp_zero = np.zeros_like(warped_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
        result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

        return result

    def slide_window(self, binary_warped, histogram):

        # Get line bases
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Define sliding window
        nwindows = 9
        window_height = np.int(binary_warped.shape[0]/nwindows)

        # Get nonzero pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Initialize sliding window
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 100
        minpix = 50
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            # Compute window coordinates
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Gather points that fit into the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append to list
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If found enough pixels, then shift the line midpoint
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Reshape index array
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Gather point coordinates
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit second orde polynoms
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Get points that fit on the polynoms
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
        left_fit[1]*nonzeroy + left_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
        right_fit[1]*nonzeroy + right_fit[2] + margin)))

        # Refit polynom
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate polynom points
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Return value
        ret = {}
        ret['leftx'] = leftx
        ret['rightx'] = rightx
        ret['left_fitx'] = left_fitx
        ret['right_fitx'] = right_fitx
        ret['ploty'] = ploty

        return ret

    def measure_curvature(self, lines_info):
        # Approximate meters per pixels
        ym_per_pix = 30/(260)
        xm_per_pix = 3.7/(650)

        # Get polynom points
        leftx = lines_info['left_fitx']
        rightx = lines_info['right_fitx']

        # Reverse order
        leftx = leftx[::-1]
        rightx = rightx[::-1]

        ploty = lines_info['ploty']

        # Compute corvature
        y_eval = np.max(ploty)
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        return left_curverad, right_curverad

    def sanity(self, ret, left_curverad, right_curverad):
        # Sanity check: whether the lines are roughly parallel and have similar curvature
        slope_left = ret['left_fitx'][0] - ret['left_fitx'][-1]
        slope_right = ret['right_fitx'][0] - ret['right_fitx'][-1]
        slope_diff = abs(slope_left - slope_right)
        slope_threshold = 150
        curve_diff = abs(left_curverad - right_curverad)
        curve_threshold = 10000

        return (slope_diff < slope_threshold and curve_diff < curve_threshold)


    def process_image(self, image):

        roi = image[450:710, 220:1150]

        # Thresholding
        binary = apply_thresholds(roi)

        # Transforming Perspective
        binary_warped, Minv = warp(binary)

        # Getting Histogram
        histogram = get_histogram(binary_warped)

        # Sliding Window to detect lane lines
        ret = self.slide_window(binary_warped, histogram)

        # Measuring Curvature
        left_curverad, right_curverad = self.measure_curvature(ret)

        # Sanity check
        if not self.sanity(ret,left_curverad,right_curverad):
            # Use last good images if sanity check fails
            binary_warped = self.used_warped
            ret = self.used_ret

        result = image

        # Visualizing Lane Lines Info
        result[450:710, 220:1150] = self.draw_lane_lines(roi, binary_warped, Minv, ret)

        # Compute deviation
        deviation_pixels = image.shape[1]/2 - abs(ret['right_fitx'][-1] - ret['left_fitx'][-1])
        xm_per_pix = 3.7/(650)
        deviation = deviation_pixels * xm_per_pix

        # update last good images
        self.used_warped = binary_warped
        self.used_ret = ret

        return result, deviation, left_curverad