from ultralytics import YOLO
import cv2
import os
import glob
import time
import torch
model_list = glob.glob("benchmark_models/*")[-1:]

def full_scale_img_test(model,N=100):
    img = cv2.imread("test.jpg")

    accum_time = 0
    for i in range(N):
        start = time.time()

        model.predict(img,verbose = False,device=0)

        accum_time += time.time()-start

    return accum_time


def rescaled_img_test(model,N=100):
    img = cv2.imread("test.jpg")

    img = cv2.resize(img,(640,640))

    accum_time = 0
    for i in range(N):
        start = time.time()

        model.predict(img,verbose = False,device=0)

        accum_time += time.time()-start

    return accum_time



tests = [("full_scale",full_scale_img_test),("rescale",rescaled_img_test)]



for model in model_list:
    N = 300
    yolo_model = YOLO(model)
    print("Benchmarking:",model)
    print("")

    img = cv2.imread("test.jpg")

    for test_name,test in tests:
        print("Performing:",test_name)
        time_taken = test(yolo_model,N)
        print("Average frame time:",time_taken/N*1000,"ms")
        print("FPS:",N/time_taken)

        print("")
        print("___________________________________")
        print("")

    del yolo_model

    torch.cuda.empty_cache()