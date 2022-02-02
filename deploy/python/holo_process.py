import cv2
import os
import numpy
import numpy as np

import sys

sys.path.append(os.path.dirname(__file__))

from infer import Predictor

cur_dir = os.path.dirname(os.path.abspath(__file__))


class HoloProcess(object):

    def __init__(self):
        self.deploy_yml = os.path.join(cur_dir, '../../output/deploy.yaml')
        self.predictor = None
        self.device = 'gpu'
        self.precision = 'int8'
        self.use_trt = False
        self.with_argmax = False
        # init predictor
        self.init_predictor()

    def init_predictor(self):
        class Config(object):

            @property
            def cfg(cself):
                return self.deploy_yml

            @property
            def print_detail(cself):
                return None

            @property
            def device(cself):
                return self.device

            @property
            def precision(cself):
                return self.precision

            @property
            def use_trt(cself):
                return self.use_trt

            @property
            def with_argmax(cself):
                return self.with_argmax

        config = Config()
        self.predictor = Predictor(config)

    def predict(self, source_filename):
        """
        Predict via segmentation network
        :param source_filename: clipped source file whose size is 512 x 512
        :return: seg_filename
        """
        return self.predictor.run_single(source_filename)

    def clip(self, source_filename):
        # image = cv2.imread(seg_filename)
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        seg_result = self.predict(source_filename)
        values = numpy.unique(seg_result)
        height, width = seg_result.shape[:2]

        for value in values:
            lower = np.array([value], dtype='uint8')
            upper = np.array([value], dtype='uint8')
            mask = cv2.inRange(seg_result, lower, upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)

                # remove background rect
                if w == width and h == height:
                    continue

                # cv2.drawContours(image, [c], 0, (0, 0, 0), 2)
                source_image = cv2.imread(source_filename)
                # for object detection, should run cv2.cvtColor(img, cv2.COLOR_BGR2RGB) first
                source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

                # clip image
                clip_image = source_image[y: y + h, x: x + w]

                yield clip_image, x, y, w, h

    def run(self, filename):
        for clip, _, _, _, _ in self.clip(filename):
            cv2.imshow('clip', clip)
            cv2.waitKey(0)


if __name__ == '__main__':
    source_filename = r"D:\Study\Github\HoloAACServer\HoloAAC\images\91662998-77fb-11ec-8ff1-70a6ccf419d3.png"
    hp = HoloProcess()
    hp.run(source_filename)
