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
    return det/2

def split_dispertion(num_current,sq,window_size=20, limit=750):
    disp = 0
    avg_sq = 0
    for i in range(num_current - window_size,num_current):
        avg_sq += sq[i]
        disp += sq[i]**2

    #avg_sq = ((avg_sq)**2)/window_size
    #disp = (disp - avg_sq)/window_size)
    #disp = np.sqrt(disp)
    disp = np.sqrt((disp - ((avg_sq)**2)/window_size)/window_size)

    print('disp = ', disp)
    #avg_sq.append(disp)
        
    if(disp < limit):
        print('added')
        return num_current
    else:
        print('out')
        return -1

def set_data(frames):
    data_frames = []
    for frame in frames:
        frame = frame.swapaxes(0,1) # swap width and height to form format W x H x C
        if len(frame.shape) < 3:
            frame = np.array([frame]).swapaxes(0,2).swapaxes(0,1) # Add grayscale channel
        data_frames.append(frame)
    frames_n = len(data_frames)
    data_frames = np.array(data_frames) # T x W x H x C
    if K.image_data_format() == 'channels_first':
        data_frames = np.rollaxis(data_frames, 3) # C x T x W x H
    return data_frames
    

def predict_videos(video_data,weight_path, absolute_max_string_len=32, output_size=28):
    if K.image_data_format() == 'channels_first':
            img_c, frames_n, img_w, img_h = video_data.shape
    else:
        frames_n, img_w, img_h, img_c = video_data.shape


    lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                    absolute_max_string_len=absolute_max_string_len, output_size=output_size)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    lipnet.model.load_weights(weight_path)

    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])

    X_data       = np.array([video_data]).astype(np.float32) / 255
    input_length = np.array([len(video_data)])

    y_pred         = lipnet.predict(X_data)

    #print(y_pred[0,0])
    #print(y_pred[0,40])
    
    result         = decoder.decode(y_pred, input_length)[0]

    return result

PREDICT_GREEDY      = False
PREDICT_BEAM_WIDTH  = 200
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'..','common','dictionaries','grid.txt')
weight_path = 'models/overlapped-weights368.h5'
face_predictor_path = os.path.join(CURRENT_PATH,'..','common','predictors','shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_predictor_path)




def stream(detector,predictor):

    #cap = cv2.VideoCapture('1.mpg')
    cap = cv2.VideoCapture(0)
    mouth_frames = []
    sq = []
    #split_nums = []
    MOUTH_WIDTH = 100
    MOUTH_HEIGHT = 50
    HORIZONTAL_PAD = 0.19
    normalize_ratio = None
    min_num = 0
    max_num = 0


    m = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if(m==224):
            break
        # detection start
        
        dets = detector(frame, 1)
        '''
        shape = None
        for k, d in enumerate(dets):
            shape = predictor(frame, d)
            i = -1
        if shape is None: # Detector doesn't detect face, just return as is
            print('Cannot detect face')
            continue

        mouth_points = []
        for part in shape.parts():
            i += 1
            if i < 48: # Only take mouth region
                continue
            mouth_points.append((part.x,part.y))
        np_mouth_points = np.array(mouth_points)
        '''
        '''
        sq.append(square_of_mouth(np_mouth_points))
        print(sq[m],m)
        m = m + 1

        if(m>20):
            max_num = split_dispertion(m,sq)
            if(max_num != -1):
                #split_nums.append(max_num)
                print(max_num)
                if(max_num - min_num > 50):
                    print('COMMAND = ',max_num)
                    video = mouth_frames[min_num:max_num]
                    #print(video)
                    video_data = set_data(video)
                    #print(video_data)
                    res = predict_videos(video_data,weight_path)
                    print('RESULT: ' , res)
                    min_num = max_num
                    
                


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
        '''
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    stream(detector,predictor)