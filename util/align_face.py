#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# this file is coming from openface: https://github.com/cmusatyalab/openface, with some changes
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import cv2
import dlib
import numpy as np
import os,errno
import random
import shutil

file_dir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(file_dir, '..', 'model')
dlib_model_dir = os.path.join(modelDir, 'dlib')

"""Module for dlib-based alignment."""
TEMPLATE = [(30, (0.7,0.5)), (36, (0.5,0.3)), (45, (0.5,0.7)),
        (39, (0.5,0.42)), (42, (0.5,0.58)), (48, (0.75, 0.38)),
        (54, (0.75,0.62))]
TEMPLATE = dict(TEMPLATE)

PATCHES = [(25,225,25,225), (25,151,25,225), (50,176,25,225), (75,201,25,225), (100,225,25,225),
        (70,180,20,160), (70,180,90,230), (120,230,55,195), (125,235,25,165), (125,235,85,225)]
PATCH_NAMES = ['g0', 'g1', 'g2', 'g3', 'g4', 'le', 're', 'nt', 'lm', 'rm']
# 0 means 31x31, 1 means 39x31, 2 means 31x39
TYPE = [1, 2, 2, 2, 2, 0, 0, 0, 0, 0]

def mkdirP(path):
    """
    Create a directory and don't error if the path already exists.

    If the directory already exists, don't do anything.

    :param path: The directory to create.
    :type path: str
    """
    assert path is not None

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class Image:
    """Object containing image metadata."""

    def __init__(self, cls, name, path):
        """
        Instantiate an 'Image' object.

        :param cls: The image's class; the name of the person.
        :type cls: str
        :param name: The image's name.
        :type name: str
        :param path: Path to the image on disk.
        :type path: str
        """
        assert cls is not None
        assert name is not None
        assert path is not None

        self.cls = cls
        self.name = name
        self.path = path

    def getBGR(self):
        """
        Load the image from disk in BGR format.

        :return: BGR image. Shape: (height, width, 3)
        :rtype: numpy.ndarray
        """
        try:
            bgr = cv2.imread(self.path)
        except:
            bgr = None
        return bgr

    def getRGB(self):
        """
        Load the image from disk in RGB format.

        :return: RGB image. Shape: (height, width, 3)
        :rtype: numpy.ndarray
        """
        bgr = self.getBGR()
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = None
        return rgb

    def __repr__(self):
        """String representation for printing."""
        return "({}, {})".format(self.cls, self.name)


def iterImgs(directory):
    u"""
    Iterate through the images in a directory.

    The images should be organized in subdirectories
    named by the image's class (who the person is)::

       $ tree directory
       person-1
       ── image-1.jpg
       ├── image-2.png
       ...
       └── image-p.png

       ...

       person-m
       ├── image-1.png
       ├── image-2.jpg
       ...
       └── image-q.png


    :param directory: The directory to iterate through.
    :type directory: str
    :return: An iterator over Image objects.
    """
    assert directory is not None

    exts = [".jpg", ".png"]

    for subdir, dirs, files in os.walk(directory):
        for path in files:
            (imageClass, fName) = (os.path.basename(subdir), path)
            (imageName, ext) = os.path.splitext(fName)
            if ext in exts:
                yield Image(imageClass, imageName, os.path.join(subdir, fName))


class AlignDlib:
    """
    Use `dlib's landmark estimation <http://blog.dlib.net/2014/08/real-time-face-pose-estimation.html>`_ to align faces.

    The alignment preprocess faces for input into a neural network.
    Faces are resized to the same size (such as 96x96) and transformed
    to make landmarks (such as the eyes and nose) appear at the same
    location on every image.

    Normalized landmarks:

    .. image:: ../images/dlib-landmark-mean.png
    """

    OUTER_EYES_AND_NOSE = [36, 45, 30]

    def __init__(self, facePredictor):
        """
        Instantiate an 'AlignDlib' object.

        :param facePredictor: The path to dlib's
        :type facePredictor: str
        """
        assert facePredictor is not None

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(facePredictor)

    def getAllFaceBoundingBoxes(self, rgbImg):
        """
        Find all face bounding boxes in an image.

        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :return: All face bounding boxes in an image.
        :rtype: dlib.rectangles
        """
        assert rgbImg is not None

        try:
            return self.detector(rgbImg, 1)
        except Exception as e:
            print("Warning: {}".format(e))
            # In rare cases, exceptions are thrown.
            return []

    def getLargestFaceBoundingBox(self, rgbImg):
        """
        Find the largest face bounding box in an image.

        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :return: The largest face bounding box in an image, or None.
        :rtype: dlib.rectangle
        """
        assert rgbImg is not None

        faces = self.getAllFaceBoundingBoxes(rgbImg)
        if len(faces) > 0:
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            return None

    def findLandmarks(self, rgbImg, bb):
        """
        Find the landmarks of a face.

        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param bb: Bounding box around the face to find landmarks for.
        :type bb: dlib.rectangle
        :return: Detected landmark locations.
        :rtype: list of (x,y) tuples
        """
        assert rgbImg is not None
        assert bb is not None

        points = self.predictor(rgbImg, bb)
        return list(map(lambda p: (p.x, p.y), points.parts()))

    def align(self, imgDim, rgbImg, bb=None, 
              landmarks=None, landmarkIndices=INNER_EYES_AND_BOTTOM_LIP, opencv_det=False, opencv_model="./model/opencv/cascade.xml"):
        r"""align(imgDim, rgbImg, bb=None, landmarks=None, landmarkIndices=INNER_EYES_AND_BOTTOM_LIP)

        Transform and align a face in an image.

        :param imgDim: The edge length in pixels of the square the image is resized to.
        :type imgDim: int
        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param bb: Bounding box around the face to align. \
                   Defaults to the largest face.
        :type bb: dlib.rectangle
        :param pad: padding bb by left, top, right, bottom
        :type pad: list
        :param landmarks: Detected landmark locations. \
                          Landmarks found on `bb` if not provided.
        :type landmarks: list of (x,y) tuples
        :param landmarkIndices: The indices to transform to.
        :type landmarkIndices: list of ints
        :return: The aligned RGB image. Shape: (imgDim, imgDim, 3)
        :rtype: numpy.ndarray
        """
        assert imgDim is not None
        assert rgbImg is not None
        assert landmarkIndices is not None

        if bb is None:
            if opencv_det:
                face_cascade = cv2.CascadeClassifier(opencv_model)
                faces = face_cascade.detectMultiScale(rgbImg, 1.1, 2, minSize=(30, 30))
                dlib_rects = []
                for (x,y,w,h) in faces:
                    dlib_rects.append(dlib.rectangle(int(x), int(y), int(x+w), int(y+h)))
                    if len(faces) > 0:
                        bb = max(dlib_rects, key=lambda rect: rect.width() * rect.height())
                    else:
                        bb = None
            else:
                bb = self.getLargestFaceBoundingBox(rgbImg)
            if bb is None:
                return
            if pad is not None:
                left = max(0, bb.left() - bb.width()*float(pad[0]))
                top = max(0, bb.top() - bb.height()*float(pad[1]))
                right = min(rgbImg.shape[1], bb.right() + bb.width()*float(pad[2]))
                bottom = min(rgbImg.shape[0], bb.bottom()+bb.height()*float(pad[3]))
                bb = dlib.rectangle(int(left), int(top), int(right), int(bottom))

        if landmarks is None:
            landmarks = self.findLandmarks(rgbImg, bb)

        npLandmarks = np.float32(landmarks)
        npLandmarkIndices = np.array(landmarkIndices)
        dstLandmarks = []
        for i in landmarkIndices:
            dstLandmarks.append(TEMPLATE[i])
        dstLandmarks = np.array(dstLandmarks) * 250

        H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices], dstLandmarks)
        thumbnail = cv2.warpAffine(rgbImg, H, (250, 250))

        return thumbnail


def write(vals, fName):
    if os.path.isfile(fName):
        print("{} exists. Backing up.".format(fName))
        os.rename(fName, "{}.bak".format(fName))
    with open(fName, 'w') as f:
        for p in vals:
            f.write(",".join(str(x) for x in p))
            f.write("\n")


def alignMain(args):
    mkdirP(args.outputDir)

    imgs = list(iterImgs(args.inputDir))

    # Shuffle so multiple versions can be run at once.
    random.shuffle(imgs)

    landmarkIndices = AlignDlib.OUTER_EYES_AND_NOSE

    align = AlignDlib(args.dlibFacePredictor)

    nFallbacks = 0
    for imgObject in imgs:
        print("=== {} ===".format(imgObject.path))
        outDir = os.path.join(args.outputDir, imgObject.cls)
        mkdirP(outDir)
        outputPrefix = os.path.join(outDir, imgObject.name)
        imgName = outputPrefix + ".png"

        if os.path.isfile(imgName):
            if args.verbose:
                print("  + Already found, skipping.")
        else:
            rgb = imgObject.getRGB()
            if rgb is None:
                if args.verbose:
                    print("  + Unable to load.")
                outRgb = None
            else:
                outRgb = align.align(args.size, rgb,
                                     landmarkIndices=landmarkIndices, opencv_det=args.opencv_det, opencv_model=args.opencv_model)
                if outRgb is None and args.verbose:
                    print("  + Unable to align.")

            if outRgb is not None:
                if args.verbose:
                    print("  + Writing aligned file to disk.")
                outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(imgName, outBgr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--inputDir', type=str, help="Input image directory.")
    parser.add_argument('--opencv-det', action='store_true', default=False,
                        help='True means using opencv model for face detection(because sometimes dlib'
                             'face detection will failed')
    parser.add_argument('--opencv-model', type=str, default='./model/opencv/cascade.xml',
                        help="Path to dlib's face predictor.")
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(dlib_model_dir, "shape_predictor_68_face_landmarks.dat"))

    parser.add_argument(
        '--outputDir', type=str, help="Output directory of aligned images.")
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    alignMain(args)
