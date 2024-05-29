import cv2
from ultralytics import YOLO
import math

class ModelAI:
    # Init
    def __int__(self):
        # VideoCapture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        # MODELS:
        # Object model
        ObjectModel = YOLO('Modelos/yolov8l.onnx')
        self.ObjectModel = ObjectModel

        # CLASES:
        # Objects
        #clsObject = ObjectModel.names
        clsObject = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
                     'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant',
                     'bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
                     'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
                     'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot',
                     'hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop',
                     'mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase',
                     'scissors','teddy bear','hair drier','toothbrush']
        self.clsObject = clsObject

        return self.cap

    # DRAW FUNCTIONS
    # Area
    def draw_area(self, img, color, xi, yi, xf, yf):
        img = cv2.rectangle(img, (xi, yi), (xf, yf), color, 1, 1)
        return img

    # Text
    def draw_text(self, img, color, text, xi, yi, size, thickness, back = False):
        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, size, thickness)
        dim = sizetext[0]
        baseline = sizetext[1]
        if back == True:
            img = cv2.rectangle(img, (xi, yi - dim[1] - baseline), (xi + dim[0], yi + baseline - 7),(0, 0, 0), cv2.FILLED)
        img = cv2.putText(img, text, (xi, yi - 5), cv2.FONT_HERSHEY_DUPLEX, size, color, thickness)
        return img

    # Line
    def draw_line(self, img, color, xi, yi, xf, yf):
        img = cv2.line(img, (xi, yi), (xf, yf), color, 1, 1)
        return img

    def area(self, frame, xi, yi, xf, yf):
        # Info
        al, an, c = frame.shape
        # Coordenates
        xi, yi = int(xi * an), int(yi * al)
        xf, yf = int(xf * an), int(yf * al)
        return xi, yi, xf, yf

    # INFERENCE
    def prediction_model(self, clean_frame, frame, model, clase):
        bbox = []
        cls = 0
        conf = 0
        # Yolo | AntiSpoof
        results = model(clean_frame, stream=True, verbose=False)
        for res in results:
            # Box
            boxes = res.boxes
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Error < 0
                if x1 < 0: x1 = 0
                if y1 < 0: y1 = 0
                if x2 < 0: x2 = 0
                if y2 < 0: y2 = 0

                bbox = [x1,y1,x2,y2]

                # Class
                cls = int(box.cls[0])

                # Confidence
                conf = math.ceil(box.conf[0])

                if clase == 0:
                    # Draw
                    objeto = self.clsObject[cls]
                    text_obj = f'{self.clsObject[cls]} {int(conf * 100)}%'

                    # Draw
                    size_obj, thickness_obj = 0.75, 1
                    frame = self.draw_text(frame, (0, 255, 0), text_obj, x1, y1, size_obj, thickness_obj, back=True)
                    frame = self.draw_area(frame, (0, 255, 0), x1, y1, x2, y2)
        return frame

    # Main
    def DetectionAI(self, cap):
        while True:
            # Frames
            ret, frame = cap.read()
            # Read keyboard
            t = cv2.waitKey(5)

            # Frame Object Detect
            clean_frame = frame.copy()

            # Areas
            # AI Scan area
            ai_area_xi, ai_area_yi, ai_area_xf, ai_area_yf = self.area(frame, 0.0351, 0.0486, 0.7539, 0.9444)
            # Draw
            color = (0,255,0)
            text_ai = f'AI Scan Area'
            size_ScanAI, thickness_ScanAI = 0.75, 1
            frame = self.draw_area(frame, color, ai_area_xi, ai_area_yi, ai_area_xf, ai_area_yf)
            frame = self.draw_text(frame, color, text_ai, ai_area_xi, ai_area_yf + 30, size_ScanAI, thickness_ScanAI)

            # Predict Object
            frame = self.prediction_model(clean_frame, frame, self.ObjectModel, clase=0)

            # Show
            cv2.imshow("Detection AI", frame)
            # Exit
            if t == 27:
                break

        # Release
        self.cap.release()
        cv2.destroyAllWindows()