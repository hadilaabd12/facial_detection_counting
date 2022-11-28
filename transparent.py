#coding=utf-8
import cv2
import numpy as np
from PIL import Image


blur_pixel = (2, 2)
move_detection_thresh = 12
face_detection_interval = 30
avg_adjustment = 0


webcam = cv2.VideoCapture(0)


width = 1280
height = 960
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


area = width * height


ret, frame = webcam.read()
avg = cv2.blur(frame, blur_pixel)
avg_float = np.float32(avg)


background = Image.open("background.jpg")


for x in range(100):
  ret, frame = webcam.read()
  if ret == False:
    break
  blur = cv2.blur(frame, blur_pixel)
  cv2.accumulateWeighted(blur, avg_float, 0.05)
  avg = cv2.convertScaleAbs(avg_float)

while(webcam.isOpened()):

  ret, frame = webcam.read()


  if ret == False:
    break


  blur = cv2.blur(frame, blur_pixel)

  
  diff = cv2.absdiff(avg, blur)

  
  gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

  
  ret, thresh = cv2.threshold(gray, move_detection_thresh, 255, cv2.THRESH_BINARY)

  
  kernel = np.ones((5, 5), np.uint8)
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

  
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
  frame[:, :, 3] = thresh
  cv2.imwrite('frame.png', frame)

  
  frame = Image.open("frame.png")
  new_frame = background.copy()
  new_frame.paste(frame, (0, 0), frame)
  new_frame.save("frame.jpg")
  reload_frame = cv2.imread('frame.jpg', cv2.IMREAD_UNCHANGED)

  
  cv2.imshow('frame', reload_frame)

  if cv2.waitKey(face_detection_interval) & 0xFF == ord('q'):
    break


  cv2.accumulateWeighted(blur, avg_float, avg_adjustment)
  avg = cv2.convertScaleAbs(avg_float)

webcam.release()
cv2.destroyAllWindows()
