





#Open google colab and create a notebook. Go to edit,notebook settings, select GPU from dropdown list.
#Run the code under snippet 1,2 and 3, one by one,after installing face_recognition library.


#	!pip install face_recognition

#Snippet 1:
#	!mkdir known
#	!wget https://www.biography.com/.image/t_share/MTY2MzU3Nzk2OTM2MjMwNTkx/elon_musk_royal_society.jpg -O known/elon.jpg
#	!wget https://www.biography.com/.image/t_share/MTE4MDAzNDEwNzg5ODI4MTEw/barack-obama-12782369-1-402.jpg -O known/obama.jpg
#	!wget https://pbs.twimg.com/profile_images/988775660163252226/XpgonN0X_400x400.jpg -O known/bill.jpg


#Snippet 2:
#	!mkdir unknown
#	!wget https://i.insider.com/5ddfa893fd9db26b8a4a2df7 -O unknown/1.jpg
#	!wget https://media.wired.com/photos/5e6c06e613205e0008da2461/master/w_2560%2Cc_limit/Biz-billgates-950211062.jpg -O unknown/2.jpg
#	!wget https://dynaimage.cdn.cnn.com/cnn/c_fill,g_auto,w_1200,h_675,ar_16:9/https%3A%2F%2Fcdn.cnn.com%2Fcnnnext%2Fdam%2Fassets%2F200516225059-02-graduate-together-obama.jpg -O unknown/3.jpg


#Snippet 3

import face_recognition
import cv2
import os
from google.colab.patches import cv2_imshow

def read_img(path):
  img=cv2.imread(path)
  (h,w)=img.shape[:2]
  width=200
  ratio=width/float(w)
  height=int(h*ratio)
  return cv2.resize(img,(width,height))

known_encodings=[]
known_names=[]
known_dir='known'

for file in os.listdir(known_dir):
  img=read_img(known_dir+'/'+file)
  img_enc=face_recognition.face_encodings(img)[0]
  known_encodings.append(img_enc)
  known_names.append(file.split('.')[0])

unknown_dir='unknown'

for file in os.listdir(unknown_dir):
  print("Processing",file)
  img=read_img(unknown_dir+'/'+file)
  img_enc=face_recognition.face_encodings(img)[0]

  results=face_recognition.compare_faces(known_encodings,img_enc)
  
  for i in range(len(results)):
    if results[i]:
      name = known_names[i]
      (top,right,bottom,left) = face_recognition.face_locations(img)[0]
      cv2.rectangle(img,(left,top),(right,bottom),(0,0,255),2) 
      cv2.putText(img,name,(left+2,bottom+20),cv2.FONT_HERSHEY_PLAIN,0.8,(0,0,255),1)
      cv2_imshow(img)