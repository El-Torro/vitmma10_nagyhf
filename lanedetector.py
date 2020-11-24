import cv2
import numpy as np

def laneDetect(img, steer_prev):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    mask_yellow = cv2.inRange(hsv, (20, 0, 100),(100, 255, 255))
    mask_white = cv2.inRange(hsv, (0, 0, 90),(255, 20, 255))
    mask = cv2.bitwise_or(mask_yellow, mask_white)
    img2 = cv2.bitwise_and(img, img, mask=mask)

    stencil = np.zeros_like(img[:,:,0])
    polygon = np.array([[0, 480], [0,400], [200,250], [440,250], [640,400], [640, 480]])
    cv2.fillConvexPoly(stencil, polygon, 1)
    img3 = cv2.bitwise_and(img2, img2, mask=stencil)
    gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img_proc = gray
    ret, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    psrc = np.float32([[290, 210], [350, 210], [140, 350], [500, 350]])
    pdst = np.float32([[200, 0], [440, 0], [190, 355], [450, 355]])
    matrix = cv2.getPerspectiveTransform(psrc, pdst)
    minv = cv2.getPerspectiveTransform(pdst, psrc)
    birdseye = cv2.warpPerspective(thresh, matrix, (w, h))

    blur = cv2.GaussianBlur(birdseye,(3, 3), 0)
    canny = cv2.Canny(blur, 40, 60)

    # Detect lines
    lines = cv2.HoughLinesP(canny, 1, np.pi/180, 30, maxLineGap=100)

    # create a copy of the original frame
    img4 = img[:,:,:].copy()

    # draw unwrapped Hough lines
    if lines is None:
        return img_proc, img4, steer_prev
    
    steer_d = steer_prev
    line_left = [0, 0, 0, 0]
    dist_left_prev = 200000
    line_right = [0, 0, 0, 0]
    dist_right_prev = 200000
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x1 = x1 - 320
        y1 = 480 - y1
        
        x2 = x2 - 320
        y2 = 480 - y2
        
        #theta = (x1 - x2) / (y1 - y2)
        distance = ( (( x2 - x1 )* (-y2))/(y2 - y1) ) - x1
        
        #left side
        if x1 < 0 and x2 < 0 :
            if line_left == [0, 0, 0, 0]:
                line_left = [x1, y1, x2, y2]
                dist_left_prev = distance
            else : 
                if dist_left_prev < distance :
                    line_left = [x1, y1, x2, y2]
                    dist_left_prev = distance
            
        if x1 > 0 and x2 > 0 :
            if line_right == [0, 0, 0, 0]:
                line_right = [x1, y1, x2, y2]
                dist_right_prev = distance
            else :
                if dist_right_prev > distance :
                    line_right = [x1, y1, x2, y2]
                    dist_right_prev = distance
    
    
    x1_left, y1_left, x2_left, y2_left = line_left
    x1_left = x1_left + 320
    y1_left = 480 - y1_left
    x2_left = x2_left + 320
    y2_left = 480 - y2_left
    
    
    x1_right, y1_right, x2_right, y2_right = line_right
    x1_right = x1_right + 320
    y1_right = 480 - y1_right
    x2_right = x2_right + 320
    y2_right = 480 - y2_right
    
    
    src_left = np.zeros((1, 1, 2))
    src_left[:, 0] = [x1_left, y1_left]
    dst_left = cv2.perspectiveTransform(src_left, minv)
    ox1_left, oy1_left = dst_left[:, 0, 0], dst_left[:, 0, 1]
    src_left[:, 0] = [x2_left, y2_left]
    dst_left = cv2.perspectiveTransform(src_left, minv)
    ox2_left, oy2_left = dst_left[:, 0, 0], dst_left[:, 0, 1]
    cv2.line(img4, (int(ox1_left), int(oy1_left)), (int(ox2_left), int(oy2_left)), (255, 0, 0), 3)
    
    src_right = np.zeros((1, 1, 2))
    src_right[:, 0] = [x1_right, y1_right]
    dst_right = cv2.perspectiveTransform(src_right, minv)
    ox1_right, oy1_right = dst_right[:, 0, 0], dst_right[:, 0, 1]
    src_right[:, 0] = [x2_right, y2_right]
    dst_right = cv2.perspectiveTransform(src_right, minv)
    ox2_right, oy2_right = dst_right[:, 0, 0], dst_right[:, 0, 1]
    cv2.line(img4, (int(ox1_right), int(oy1_right)), (int(ox2_right), int(oy2_right)), (255, 0, 0), 3)
    
    
    x_dest = dist_left_prev / dist_right_prev
    if x_dest < -0.5 and x_dest > -1.5:
        x_dest = -1
    
    x_d = (dist_right_prev + dist_left_prev) / 2
    

    if dist_left_prev != 200000 and dist_right_prev != 200000 :
#        steer_d = -x_dest / 320
        steer_d = (x_dest + 1) * 0.15
    

    if steer_d > 0.3:
        steer_d = 0.3
    
    if steer_d < -0.3:
        steer_d = -0.3
    
    print(dist_left_prev, dist_right_prev, steer_d)
    
    return img_proc, img4, steer_d
