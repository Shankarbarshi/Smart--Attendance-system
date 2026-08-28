import cv2, os, csv
from datetime import datetime
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

path = 'images'
faces, ids, names, id_map = [], [], [], {}
curr = 0
for f in os.listdir(path):
    if f.lower().endswith(('.jpg','.png','.jpeg')):
        name = os.path.splitext(f)[0]
        if name not in id_map:
            id_map[name]=curr; curr+=1; names.append(name)
        img = cv2.imread(os.path.join(path,f), cv2.IMREAD_GRAYSCALE)
        for (x,y,w,h) in face_cascade.detectMultiScale(img,1.1,4):
            faces.append(img[y:y+h,x:x+w]); ids.append(id_map[name])

if faces:
    recognizer.train(faces, np.array(ids))
    print(f"Training done: {names}")
else:
    print("images folder madhe face nahi sapadla")

cap = cv2.VideoCapture(0)
marked=set()
def mark(name):
    with open('attendance.csv','a',newline='') as f:
        csv.writer(f).writerow([name, datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%H:%M:%S")])
    print(f"Present: {name}")

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for (x,y,w,h) in face_cascade.detectMultiScale(gray,1.1,4):
        id_,conf = recognizer.predict(gray[y:y+h,x:x+w])
        if conf < 70:
            name=names[id_]
            if name not in marked:
                mark(name); marked.add(name)
            col=(0,255,0)
        else:
            name="Unknown"; col=(0,0,255)
        cv2.rectangle(frame,(x,y),(x+w,y+h),col,2)
        cv2.putText(frame,f"{name} {int(conf)}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,col,2)
    cv2.imshow('MIT Barshi - Smart Attendance (q to close)',frame)
    if cv2.waitKey(1)==ord('q'): break
cap.release(); cv2.destroyAllWindows()