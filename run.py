import cv2
import numpy as np
import av
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

# Load YOLO
net = cv2.dnn.readNet("yolov3-320.weights", "yolov3-320.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
try:
    # For OpenCV >= 4.5.3
    output_layer_indexes = net.getUnconnectedOutLayers().flatten()
except AttributeError:
    # For older OpenCV versions
    output_layer_indexes = net.getUnconnectedOutLayers()

output_layers = [layer_names[i - 1] for i in output_layer_indexes]


class YOLOVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)  # Adjusted to 320x320
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Confidence threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indexes) > 0:  # Debugging output
            print(f"Detected objects: {len(indexes)}")
        else:
            print("No objects detected")

        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit-WebRTC streamer
webrtc_streamer(key="example", video_processor_factory=YOLOVideoProcessor)
