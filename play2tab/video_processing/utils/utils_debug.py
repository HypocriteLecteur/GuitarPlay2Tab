import cv2

class VideoBuffer:
    def __init__(self, buffer=15) -> None:
        self.buffer = buffer
        self.data = []
    
    def put(self, image):
        self.data.insert(0, image)
        if len(self.data) > self.buffer:
            self.data.pop()
    
    def imshow(self):
        cv2.imshow('videoplayback', self.data[0])
        key = cv2.waitKey(10)
        if key == ord('p'):
            self.playback()
    
    def playback(self):
        index = 0
        while True:
            cv2.imshow('videoplayback', self.data[index])
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
            elif key == ord('a'):
                if index < self.buffer-1:
                    index = index + 1
            elif key == ord('d'):
                if index > 0:
                    index = index - 1