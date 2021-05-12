# -*- coding: utf-8 -*-
import pandas
import numpy
import cv2
import time
from datetime import datetime
import logging

logger = logging.getLogger()


class CNN:
    def __init__(self):
        self.path_model = './model/model.h5'

    @staticmethod
    def __load_model(self):
        self.model = ''

    def run(self, img):
        out = self.model.predict(img)
        return out


class FocusOnDriving:
    def __init__(self, model):
        self.model = model
        self.url = 0

    def run(self):
        self.__connect_cam()
        out = None
        while self.cam.isOpened():
            frame, timestamp, error = self.get_image()
            if error:
                break
            # out = self.model.run(frame)
            print(f'{datetime.fromtimestamp(timestamp)} | {out}')

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def __connect_cam(self):
        self.cam = cv2.VideoCapture(self.url)

    def reset(self):
        self.cam.release()

    def get_image(self):
        error = False
        count = 0
        for i in range(10):
            self.cam.grab()
        ret, frame = self.cam.read()
        if ret:
            count = 0
        else:
            count += 1
            self.reset()
            self.__connect_cam()
            time.sleep(5)
            self.get_image()
            if count > 10:
                error = True

        return frame, time.time(), error

    def show(self, img):
        cv2.imshow('img', img)
        cv2.waitKey()


if __name__ == '__main__':
    focus = FocusOnDriving(model=CNN())
    focus.run()
