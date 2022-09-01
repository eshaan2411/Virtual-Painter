import cv2
import mediapipe as mp
import time


class HandDetect():
    def __init__(self, 
                mode = False, 
                max_hands=2, 
                modelComplexity=1,
                detectionConfidence=0.5,
                trackConfidence=0.5):
        
        self.mode = mode
        self.max_hands = max_hands
        self.modelComplexity = modelComplexity
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.max_hands, self.modelComplexity,
                                        self.detectionConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        # ID = [thumb, index, middle, ring, pinky]
        self.fingerTipId = [4, 8, 12, 16, 20]

    def getHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
    
        if self.results.multi_hand_landmarks:
            for each_hand in self.results.multi_hand_landmarks:
                if draw==True:      
                    self.mpDraw.draw_landmarks(img, each_hand, self.mphands.HAND_CONNECTIONS)
        return img

    def getPosition(self, img, handNo=0, draw=True):
        
        self.landmarkList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) 
                self.landmarkList.append([id, cx, cy])

                if id==4 or id==8 or id==12 or id==16 or id==20:
                    if draw==True:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return self.landmarkList

    def getFingersUp(self):
        fingers = []
        
        # Checking for thumb
        if self.landmarkList[4][1] < self.landmarkList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Checking for other 4 fingers
        for id in range(1, 5):
            if self.landmarkList[self.fingerTipId[id]][2] < self.landmarkList[self.fingerTipId[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers


def main():
    previous = 0
    current = 0
    cap = cv2.VideoCapture(0)

    detector = HandDetect()
    while True:
        sucess, img = cap.read()
        img = detector.getHands(img)
        landmarkList = detector.getPosition(img)
        if len(landmarkList)!=0:
            print(landmarkList)

        current = time.time()
        fps = 1/(current-previous)
        previous = current
    
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()