import sys
from threading import Thread

import cv2
import imutils
import numpy as np

from yoloDet import YoloTRT
import time

result = []


def PreProcessImg(input_w, input_h, img):
    image_raw = img
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    r_w = input_w / w
    r_h = input_h / h
    if r_h > r_w:
        tw = input_w
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((input_h - th) / 2)
        ty2 = input_h - th - ty1
    else:
        tw = int(r_h * w)
        th = input_h
        tx1 = int((input_w - tw) / 2)
        tx2 = input_w - tw - tx1
        ty1 = ty2 = 0
    image = cv2.resize(image, (tw, th))
    image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128))
    image = image.astype(np.float32)
    image /= 255.0
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = np.ascontiguousarray(image)
    if len(result) == 0:
        result.append((image, image_raw, h, w))
    else:
        result[0] = (image, image_raw, h, w)


# use path for library and engine file
models = []
num_models = 1
for i in range(num_models):
    models.append(YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build_s/yolov5s.engine", conf=0.5,
                          yolo_ver="v5"))
cap = cv2.VideoCapture("videos/testvideo.mp4")

input_w = models[0].input_w
input_h = models[0].input_h
first = True
avg = 0
i = 0
while True:
    i = i + 1
    if first:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        thread = Thread(target=PreProcessImg, args=(input_w, input_h, frame))
        thread.start()
        first = False

    start = time.time()
    thread.join()
    (image, image_raw, h, w) = result[0]
    detections, t = models[i % num_models].Inference(image, image_raw, h, w)

    if i % 20 == 0:
        models[0] = YoloTRT(library="yolov5/build/libmyplugins.so",
                            engine="yolov5/build_s/yolov5s.engine", conf=0.5,
                            yolo_ver="v5")

    # for i in range(5):
    #     for m in models:
    #         detections, t = m.Inference(input_image, image_raw, origin_h, origin_w)

    # for obj in detections:
    #    print(obj['class'], obj['conf'], obj['box'])
    # print("FPS: {} sec".format(1/t))
    # cv2.imshow("Output", frame)
    # key = cv2.waitKey(1)
    # if key == ord('q'):
    #     break
    end = time.time()
    avg = 0.9 * avg + (end - start) * 0.1
    print("{} sec".format(end - start))
    # create a thread
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    thread = Thread(target=PreProcessImg, args=(input_w, input_h, frame))
    thread.start()

cap.release()
cv2.destroyAllWindows()
