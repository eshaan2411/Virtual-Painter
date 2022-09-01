import numpy as np
import cv2
import os
import hand_tracking as ht
from flask import Flask, render_template
from flask import Response
from flask import url_for

app = Flask(__name__)

global cap
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

def generate_frames():
    folder = "Headers"
    mylist = os.listdir(folder)
    overlayList = list()

    # overlayList = [Purple, Light blue, Pink, Green, Yellow, Orange, Cyan, Red]
    for imgpath in mylist:
        image = cv2.imread(f"{folder}/{imgpath}")
        overlayList.append(image)

    header = overlayList[0]
    drawColor = (255, 102, 178)
    brushThickness = 14
    eraserThickness = 50
    xp, yp = 0, 0

    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
    detector = ht.HandDetect(detectionConfidence=0.85)

    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            img = cv2.flip(img, 1)
            img = detector.getHands(img)
            landmarkList = detector.getPosition(img, draw=False)
            
            if len(landmarkList)!=0:                        
                # index finger tip - (x1, y1)
                x1, y1 = landmarkList[8][1:]
                # middle finger tip - (x2, y2)
                x2, y2 = landmarkList[12][1:]

                # Check the fingers 
                fingers = detector.getFingersUp()

                # Two fingers - Select mode
                if fingers[1] and fingers[2]:
                    xp, yp = 0, 0

                    if y1 < 125:
                        if 130<x1<265:
                            header = overlayList[0]
                            drawColor = (255, 102, 178)
                        elif 265<x1<380:
                            header = overlayList[1]
                            drawColor = (255, 255, 0)
                        elif 380<x1<500:
                            header = overlayList[2]
                            drawColor = (153, 51, 255)
                        elif 500<x1<615:
                            header = overlayList[3]
                            drawColor = (0, 153, 0)
                        elif 615<x1<735:
                            header = overlayList[4]
                            drawColor = (51, 255, 255)
                        elif 735<x1<850:
                            header = overlayList[5]
                            drawColor = (51, 153, 255)
                        elif 850<x1<970:
                            header = overlayList[6]
                            drawColor = (153, 255, 204)
                        elif 970<x1<1090:
                            header = overlayList[7]
                            drawColor = (0, 0, 255)
                        elif 1090<x1:
                            header = overlayList[8]
                            drawColor = (0, 0, 0)

                # One finger - Draw mode
                if fingers[1] and fingers[2]==False:
                    cv2.circle(img, (x1, y1), 14, drawColor, cv2.FILLED)

                    if xp==0 and yp==0:
                        xp, yp = x1, y1

                    if drawColor == (0, 0, 0):
                        cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    else:
                        cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                    
                    xp, yp = x1, y1

            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            
            img = cv2.bitwise_and(img, imgInv)
            img = cv2.bitwise_or(img, imgCanvas)

            # Set the header image
            img[0:125, 0:1280] = cv2.resize(header, (1280, 125))  
            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/begin')
def begin():
    return render_template("begin.html")

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug = True)

cap.release()
cv2.destroyAllWindows()