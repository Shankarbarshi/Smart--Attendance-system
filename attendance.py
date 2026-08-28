import cv2, os, csv
from datetime import datetime
import numpy as np
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
faces,ids,names,id_map=[],[],[],{}
c=0
for f in os.listdir('images'):
 if f.lower().endswith(('.jpg','.jpeg','.png')):
  name=os.path.splitext(f)[0].split('.')[0]
  if name not in id_map:
   id_map[name]=c;c+=1;names.append(name)
  img=cv2.imread(os.path.join('images',f),0)
  for (x,y,w,h) in face_cascade.detectMultiScale(img,1.1,4):
   faces.append(img[y:y+h,x:x+w]);ids.append(id_map[name])
recognizer.train(faces,np.array(ids))
print("Training zali:",names)
cap=cv2.VideoCapture(0)
done=[]
while True:
 ret,fr=cap.read()
 g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
 for (x,y,w,h) in face_cascade.detectMultiScale(g,1.1,4):
  id_,co=recognizer.predict(g[y:y+h,x:x+w])
  if co<75:
   nm=names[id_]
   col=(0,255,0)
   if nm not in done:
    open('attendance.csv','a',newline='').write(f"{nm},{datetime.now()}\n")
    print(f"Present: {nm}");done.append(nm)
  else:
   nm="Unknown";col=(0,0,255)
  cv2.rectangle(fr,(x,y),(x+w,y+h),col,2)
  cv2.putText(fr,nm,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,col,2)
 cv2.imshow("MIT Barshi - Q to close",fr)
 if cv2.waitKey(1)==ord('q'):break
cap.release();cv2.destroyAllWindows()