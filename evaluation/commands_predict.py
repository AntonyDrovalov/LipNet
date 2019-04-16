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

np.random.seed(55)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

FACE_PREDICTOR_PATH = os.path.join(CURRENT_PATH,'..','common','predictors','shape_predictor_68_face_landmarks.dat')

PREDICT_GREEDY      = False
PREDICT_BEAM_WIDTH  = 200
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'..','common','dictionaries','grid.txt')

def process_video(weight_path,video_path):
    print "\nLoading data from disk..."
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    if os.path.isfile(video_path):
        video.from_video(video_path)
    else:
        video.from_frames(video_path)
    print "Data loaded.\n"

    a = video.split_commands()
    show_square(video.sq[20:],video.avg_sq)
    
    ans_v = []
    ans_r = []

    if (a!=[]):
        for i in range(len(a)):
            if( i == 0):
                video.from_video_test(video_path,0,a[i])
                v,r = predict_videos(video,weight_path)
                ans_v.append(v)
                ans_r.append(r)

            if(i==len(a)-1):
                video.from_video_test(video_path,a[i],-1,last=True)
                v,r = predict_videos(video,weight_path)
                ans_v.append(v)
                ans_r.append(r)
                break

            video.from_video_test(video_path,a[i],a[i+1])
            v,r = predict_videos(video,weight_path)
            ans_v.append(v)
            ans_r.append(r)
    return ans_v,ans_r

def predict_videos(video,weight_path, absolute_max_string_len=32, output_size=28):
    if K.image_data_format() == 'channels_first':
            img_c, frames_n, img_w, img_h = video.data.shape
    else:
        frames_n, img_w, img_h, img_c = video.data.shape


    lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                    absolute_max_string_len=absolute_max_string_len, output_size=output_size)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    lipnet.model.load_weights(weight_path)

    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])

    X_data       = np.array([video.data]).astype(np.float32) / 255
    input_length = np.array([len(video.data)])

    y_pred         = lipnet.predict(X_data)
    result         = decoder.decode(y_pred, input_length)[0]

    return (video, result)


def predict(weight_path, video_path, absolute_max_string_len=32, output_size=28):
    print "\nLoading data from disk..."
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    if os.path.isfile(video_path):
        video.from_video(video_path)
    else:
        video.from_frames(video_path)
    print "Data loaded.\n"

    a = video.split_commands()
    show_square(video.sq[20:],video.avg_sq)
    
    if (a!=[]):
        for i in range(len(a)):
            if(i==len(a)-1):
                a[i+1] = len(a)
            video.from_video_test(video_path,a[i],a[i+1])


        if K.image_data_format() == 'channels_first':
            img_c, frames_n, img_w, img_h = video.data.shape
        else:
            frames_n, img_w, img_h, img_c = video.data.shape


        lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                    absolute_max_string_len=absolute_max_string_len, output_size=output_size)

        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
        lipnet.model.load_weights(weight_path)

        spell = Spell(path=PREDICT_DICTIONARY)
        decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])

        X_data       = np.array([video.data]).astype(np.float32) / 255
        input_length = np.array([len(video.data)])

        y_pred         = lipnet.predict(X_data)
        result         = decoder.decode(y_pred, input_length)[0]

    return (video, result)

if __name__ == '__main__':
    video = []
    result = []
    video,result = process_video(sys.argv[1], sys.argv[2])

    list_v_r = zip(video,result)

    for item in list_v_r:
        if item[0] is not None:
            show_video_subtitle(item[0].face, item[1])

            stripe = "-" * len(item[1])
    
            print ""
            print "             --{}- ".format(stripe)
            print "[ DECODED ] |> {} |".format(item[1])
            print "             --{}- ".format(stripe)
    '''
    for v,r in video,result:
        if v is not None:
            #show_video_subtitle(v.face, r)

            stripe = "-" * len(r)
    
            print ""
            print "             --{}- ".format(stripe)
            print "[ DECODED ] |> {} |".format(r)
            print "             --{}- ".format(stripe)

    
    if len(sys.argv) == 3:
        video, result = predict(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        video, result = predict(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5:
        video, result = predict(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        video, result = None, ""

    if video is not None:
        show_video_subtitle(video.face, result)

    stripe = "-" * len(result)
    print ""
    print " __                   __  __          __      "
    print "/\\ \\       __        /\\ \\/\\ \\        /\\ \\__   "
    print "\\ \\ \\     /\\_\\  _____\\ \\ `\\\\ \\     __\\ \\ ,_\\  "
    print " \\ \\ \\  __\\/\\ \\/\\ '__`\\ \\ , ` \\  /'__`\\ \\ \\/  "
    print "  \\ \\ \\L\\ \\\\ \\ \\ \\ \\L\\ \\ \\ \\`\\ \\/\\  __/\\ \\ \\_ "
    print "   \\ \\____/ \\ \\_\\ \\ ,__/\\ \\_\\ \\_\\ \\____\\\\ \\__\\"
    print "    \\/___/   \\/_/\\ \\ \\/  \\/_/\\/_/\\/____/ \\/__/"
    print "                  \\ \\_\\                       "
    print "                   \\/_/                       "
    print ""
    print "             --{}- ".format(stripe)
    print "[ DECODED ] |> {} |".format(result)
    print "             --{}- ".format(stripe)
    '''