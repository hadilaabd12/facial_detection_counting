#coding=utf-8
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2


interval = 200
specific_object_name = "person"
specific_object_minimum_confidence = 0.75


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


vs = VideoStream(src=0).start()

time.sleep(2.0)

fps = FPS().start()


while True:
	
	frame = vs.read()
	if len(frame) > 0:
		if len(frame[0]) > 0:
			frame = imutils.resize(frame, width=720)

			
			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
				0.007843, (300, 300), 127.5)

			
			net.setInput(blob)
			detections = net.forward()

			frameList = []

			
			for i in np.arange(0, detections.shape[2]):
				
				confidence = detections[0, 0, i, 2]

				
				if confidence > args["confidence"]:
					
					idx = int(detections[0, 0, i, 1])
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					
					label_name = CLASSES[idx]
					label_confidence = confidence
					if label_name == specific_object_name and label_confidence > specific_object_minimum_confidence:
						
						if endY > startY and endX > startX:
							frameList.append(frame[startY:endY, startX:endX])
						elif startY > endY and startX > endX:
							frameList.append(frame[endY:startY, endX:startX])
						
						label = "{}: {:.2f}%".format(CLASSES[idx],
							confidence * 100)
						cv2.rectangle(frame, (startX, startY), (endX, endY),
							COLORS[idx], 2)
						y = startY - 15 if startY - 15 > 15 else startY + 15
						cv2.putText(frame, label, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

			
			cv2.imshow("Frame", frame)
			frameIndex = 1
			for _frame in frameList:
				if len(_frame) > 0:
					if len(_frame[0]) > 0:
						_frame = cv2.resize(_frame, (240, 240), interpolation=cv2.INTER_CUBIC)
						cv2.imshow("Frame"+str(frameIndex), _frame)
						frameIndex = frameIndex + 1

			if cv2.waitKey(interval) & 0xFF == ord('q'):
				break

			
			fps.update()


fps.stop()

cv2.destroyAllWindows()
vs.stop()