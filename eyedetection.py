import cv2,glob

all_img=glob.glob("*.jpg")

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier("haascascade_eye.xml")

for image in all_img:

    img=cv2.imread(image)
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray_img,1.3,5)

    for (x,y,w,h) in faces:

        final_img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
        roi_gray=gray_img[y:y+h, x:x+w]
        roi_color=img[y:y+h, x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray)

        for(ex,ey,ew,eh) in eyes:
            yes=cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),3)



    cv2.imshow("eye detection",yes)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
