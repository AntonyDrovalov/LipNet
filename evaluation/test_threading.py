import threading
import cv2
import time
import dlib

class VideoCaptureAsync:
    def __init__(self, src=0, width=640, height=480, nframes=120):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.nframes = nframes
        self.frames = []
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.capset.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        n = 0
        while self.started:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.read_lock:
                    self.frames.append(frame)
                n+=1
                if n>=self.nframes: break
        self.started = False

    def read(self,n):
        frame = None
        with self.read_lock:
            if n<len(self.frames): frame = self.frames[n]
        return frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()


def test(n_frames=500, width=1280, height=720, async=False):
    detector = dlib.get_frontal_face_detector()
    if async:
        cap = VideoCaptureAsync(0)
    else:
        cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if async:
        cap.start()
    t0 = time.time()
    i = 0
    while i < n_frames:
        _, frame = cap.read()
        #dets = detector(frame, 1)
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)
        i += 1
    t1 = time.time() - t0
    print('[i] Frames per second: {:.2f} {}, async={}'.format(n_frames / t1, t1, async))
    if async:
        cap.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test(n_frames=500, width=640, height=480, async=False)
    test(n_frames=500, width=640, height=480, async=True)