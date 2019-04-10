from lipnet.lipreading.videos import Video
from lipnet.lipreading.visualization import show_video_subtitle,show_square,show_sq_audio
import numpy as np
import sys
import os

np.random.seed(55)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

FACE_PREDICTOR_PATH = os.path.join(CURRENT_PATH,'..','common','predictors','shape_predictor_68_face_landmarks.dat')



def predict(video_path):
    print "\nLoading data from disk..."
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    if os.path.isfile(video_path):
        video.from_video(video_path)
    else:
        video.from_frames(video_path)
    print "Data loaded.\n"

    a = video.split_commands()

    avg = 0
    for item in video.sq:
        avg += item
    avg = avg/len(video.sq)
    print('Avarage square = ', avg) 

    show_square(video.sq[20:],video.avg_sq)
    #show_square(video.avg_sq)

    #show_sq_audio(video.sq,'/home/anton/LipNet/evaluation/samples/test.wav')

    
    
    #video.from_video_test(video_path,a)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        predict(sys.argv[1])
    

    