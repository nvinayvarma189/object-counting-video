# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
from pathlib import Path
from sort import Sort
from typing import Dict, List
import json
import fire


class ObjectCounterVideo:
    # Class attributes
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
    video_writer = None
    memory: Dict = {}

    def __init__(
        self,
        model_labels_path="./yolo-coco/coco.names",
        model_weights_path="./yolo-coco/yolov3.weights",
        model_config_path="./yolo-coco/yolov3.cfg",
        input_video_path="./input/highway.mp4",
        output_video_path="./output/highway.avi",
        output_json_path="./output/highway_objects.json",
        confidence_threshold=0.5,
        nms_threshold=0.1,
        objects_to_count=["car", "bus", "motorbike", "bicycle", "truck"],
        line_coords=[(43, 543), (550, 655)],
        line_color=(0, 255, 255),
        line_width=5,
    ):
        self.labels = open(model_labels_path).read().strip().split("\n")
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.line_color = line_color
        self.line_width = line_width
        self.line_coords = line_coords
        self.video_capture = cv2.VideoCapture(input_video_path)
        self.output_video_path = output_video_path
        self.output_json_path = output_json_path
        self.instances_count = self.initialize_instance_counter(objects_to_count)
        self.tracker = Sort()
        self.net, self.ln = self.load_model(model_config_path, model_weights_path)

    @staticmethod
    def load_model(model_config_path, model_weights_path):
        net = cv2.dnn.readNetFromDarknet(model_config_path, model_weights_path)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return net, ln

    @staticmethod
    def initialize_instance_counter(objects_to_count):
        instances_count = {}
        for obj in objects_to_count:
            instances_count[obj] = 0
        return instances_count

    @staticmethod
    def _ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _is_object_box_intersecting_with_line(self, A, B, C, D):
        return self._ccw(A, C, D) != self._ccw(B, C, D) and self._ccw(
            A, B, C
        ) != self._ccw(A, B, D)

    def determine_number_of_frames_in_video(self):
        try:
            prop = (
                cv2.cv.CV_CAP_PROP_FRAME_COUNT
                if imutils.is_cv2()
                else cv2.CAP_PROP_FRAME_COUNT
            )
            total_num_frames = int(self.video_capture.get(prop))
            print("[INFO] Found {} number of frames in video".format(total_num_frames))
        except Exception as e:
            print(e)
            print("[INFO] could not determine # of frames in video")
            print("[INFO] no approx. completion time can be provided")
            total_num_frames = -1
        return total_num_frames

    def count_objects_in_video(self):
        frame_index = 0
        W, H = (None, None)
        self.determine_number_of_frames_in_video()

        while True:
            ret, frame = self.video_capture.read()

            # end of video. No more frames to grab
            if not ret:
                break
            # if the frame dimensions are empty, grab them
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            blob_img = cv2.dnn.blobFromImage(
                frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
            )
            self.net.setInput(blob_img)
            layer_outputs = self.net.forward(self.ln)

            # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
            boxes = []
            confidences = []
            class_ids = []

            # loop over each of the layer outputs
            for detections in layer_outputs:
                # loop over each of the detections
                for detection in detections:
                    # extract the class ID and confidence (i.e., probability)
                    # of the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > self.confidence_threshold:
                        # scale the bounding box coordinates back relative to
                        # the size of the image, keeping in mind that YOLO
                        # actually returns the center (x, y)-coordinates of
                        # the bounding box followed by the boxes' width and
                        # height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top
                        # and and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # update our list of bounding box coordinates,
                        # confidences, and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping bounding boxes
            nms_boxes = cv2.dnn.NMSBoxes(
                boxes, confidences, self.confidence_threshold, self.nms_threshold
            )

            dets = []
            if len(nms_boxes) > 0:
                # loop over the indexes we are keeping
                for i in nms_boxes.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    dets.append([x, y, x + w, y + h, confidences[i]])

            np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
            dets = np.asarray(dets)
            tracks = self.tracker.update(dets)

            tracked_boxes = []
            index_ids = []
            previous_box_detections = self.memory.copy()
            self.memory = {}

            for track in tracks:
                tracked_boxes.append([track[0], track[1], track[2], track[3]])
                index_ids.append(int(track[4]))
                self.memory[index_ids[-1]] = tracked_boxes[-1]

            # print(len(previous_box_detections))
            if len(tracked_boxes) > 0:
                i = int(0)
                for box in tracked_boxes:
                    # extract the bounding box coordinates
                    (x, y) = (int(box[0]), int(box[1]))
                    (w, h) = (int(box[2]), int(box[3]))

                    color = [
                        int(c) for c in self.COLORS[index_ids[i] % len(self.COLORS)]
                    ]
                    cv2.rectangle(frame, (x, y), (w, h), color, 2)

                    if index_ids[i] in previous_box_detections:
                        previous_box = previous_box_detections[index_ids[i]]
                        (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                        (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                        p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                        p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                        cv2.line(frame, p0, p1, color, 3)
                        if self._is_object_box_intersecting_with_line(
                            p0, p1, self.line_coords[0], self.line_coords[1]
                        ):
                            self.instances_count[self.labels[class_ids[i]]] += 1

                    text = "{}".format(index_ids[i])
                    cv2.putText(
                        frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
                    i += 1

            # draw line
            cv2.line(
                frame,
                self.line_coords[0],
                self.line_coords[1],
                self.line_color,
                self.line_width,
            )

            # draw counter
            for idx, (key, value) in enumerate(self.instances_count.items()):
                cv2.putText(
                    frame,
                    f"# of {key}: {self.instances_count[key]}",
                    (30, 50 + idx * 40),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

            # check if the video writer is None
            if self.video_writer is None:
                # initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.video_writer = cv2.VideoWriter(
                    self.output_video_path,
                    fourcc,
                    30,
                    (frame.shape[1], frame.shape[0]),
                    True,
                )

            # write the output frame to disk
            self.video_writer.write(frame)
            frame_index += 1

        with open(self.output_json_path, "w") as fp:
            json.dump(self.instances_count, fp)

        print("[INFO] Finshed writing video...")
        self.video_writer.release()
        self.video_capture.release()


if __name__ == "__main__":
    fire.Fire(ObjectCounterVideo)
