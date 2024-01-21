import cv2

net = cv2.dnn.readNet("model/yolov4-tiny.weights","model/yolov4-tiny.cfg")

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320),scale=1/255)
# model.setInputParams(size=(416,416), scale=1/255)


# Load Classes
classes = []
with open("model/labels.txt","r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

# Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



while True:
    # Frames
    ret, frame = cap.read()
    # Object detection
    class_ids, scores, bboxes = model.detect(frame)
    for class_id, score, bbox in zip(class_ids,scores,bboxes):
        x, y, w, h = bbox
        cv2.putText(frame, str(classes[class_id]), (x,y - 10) ,cv2.FONT_HERSHEY_PLAIN, 2, (200,0,50),2)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (200,0,50), 3)
    print("class_id", class_ids)
    print("scores", scores)
    
    cv2.imshow("Fram", frame)
    cv2.waitKey(1)