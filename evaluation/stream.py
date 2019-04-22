import numpy as np
import cv2
import time
from lipnet.lipreading.videos import Video
from lipnet.lipreading.visualization import show_video_subtitle,show_square
from lipnet.core.decoders import Decoder
from lipnet.lipreading.helpers import labels_to_text
from lipnet.utils.spell import Spell
from lipnet.model2 import LipNet
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import sys
import os
import dlib
from scipy.misc import imresize

def square_of_mouth(points):
    det = 0
    l = len(points)
    for i in range(l-1):
        x1 = [points[i,0],points[i,1]]
        x2 = [points[i+1,0],points[i+1,1]]
        a = [x1,x2]
        det += abs(np.linalg.det(a))
    x1 = [points[l-1,0],points[l-1,1]]
    x2 = [points[0,0],points[0,1]]
    det += abs(np.linalg.det(a))
    det = det/2
    return det

def split_dispertion(num_current,sq,window_size=20, limit=750):
    disp = 0
    avg_sq = 0
    for i in range(num_current - window_size,num_current):
        avg_sq += sq[i]
        disp += sq[i]**2

    avg_sq = ((avg_sq)**2)/window_size
    disp = (disp - avg_sq)/window_size
    disp = np.sqrt(disp)

    print('disp = ', disp)
    #avg_sq.append(disp)
        
    if(disp < limit):
        print('added')
        return num_current
    else:
        print('out')
        return -1

def split_commands(split_nums):
    #if(not self.split_nums):
    #    return frames
    #else:
    points = []
    min_num = 0
    max_num = split_nums[0]
    for i in range(len(split_nums)):
        print('num: ',split_nums[i])
        #print('min num: ', min_num)
        #print('max num: ', max_num)
        if(max_num - min_num > 50):
            points.append(split_nums[i])
            print('command: ',split_nums[i])
            min_num = split_nums[i]
        else:
            max_num = split_nums[i]
        
        if(i==len(split_nums)-1):
            if(max_num - min_num > 50):
                points.append(split_nums[i])
                print('command: ',split_nums[i])
    return points

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
face_predictor_path = os.path.join(CURRENT_PATH,'..','common','predictors','shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_predictor_path)



def stream(detector,predictor):

    #cap = cv2.VideoCapture('1.mpg')
    cap = cv2.VideoCapture(0)
    mouth_frames = []
    sq = []
    split_nums = []
    MOUTH_WIDTH = 100
    MOUTH_HEIGHT = 50
    HORIZONTAL_PAD = 0.19
    normalize_ratio = None

    m = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(m==224):
            break
        # detection start
        dets = detector(frame, 1)
        shape = None
        for k, d in enumerate(dets):
            shape = predictor(frame, d)
            i = -1
        if shape is None: # Detector doesn't detect face, just return as is
            print('Cannot detect face')

        mouth_points = []
        for part in shape.parts():
            i += 1
            if i < 48: # Only take mouth region
                continue
            mouth_points.append((part.x,part.y))
        np_mouth_points = np.array(mouth_points)

        sq.append(square_of_mouth(np_mouth_points))
        print(sq[m],m)
        m = m + 1

        if(m>20):
            split_frame = split_dispertion(m,sq)
            if(split_frame != -1):
                split_nums.append(split_frame)
                print(split_frame)

        mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

        if normalize_ratio is None:
            mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
            mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

            normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

        new_img_shape = (int(frame.shape[0] * normalize_ratio   ), int(frame.shape[1] * normalize_ratio))
        resized_img = imresize(frame, new_img_shape)

        mouth_centroid_norm = mouth_centroid * normalize_ratio

        mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
        mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
        mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
        mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

        mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]

        mouth_frames.append(mouth_crop_image)
        # detection end





        mouth_drawed_image = frame
        cv2.circle(mouth_drawed_image,(int(mouth_centroid[0]),int(mouth_centroid[1])),10,(0,0,0))
        for item in np_mouth_points:
            cv2.circle(mouth_drawed_image,(int(item[0]),int(item[1])),5,(0,255,0))
        cv2.imshow('frame',mouth_drawed_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    split_commands(split_nums)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    stream(detector,predictor)