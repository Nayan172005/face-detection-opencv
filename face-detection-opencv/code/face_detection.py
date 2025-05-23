import cv2  #computer vision
import cv2.data
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  #loading pretrained model 
cap=cv2.VideoCapture(0)  #type 1 or 2 in case of external webcam in paranthesis, 0 is in case of internal cam
while True:
    ret,frame=cap.read()    #cap.read() return two values, ret contains bool value which is true if webcam is working else false, frame is getting the actual footage from the webcam
    if not ret:    #checking if the webcam is working properly or not
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   #converting to grayscale

    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))   #passed the grayscale image, reduce size of the image by 10%, if minimum 5 detectors day its a image then only it will be detected as a image, if anything is smaller than 30 by 30px then it wont be detected.

    #now face detection is done, we now want to draw box around the face
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)    #dimensions, color, width
    cv2.imshow('face_detector',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):      #cv2.waitKey(1) checks if the q key is pressed for more than 1 milisecond or not, 0xFF masks in hexadecimal which makes it possible to use the program in various other systems.
        break

cap.release()
cv2.destroyAllWindows()
    