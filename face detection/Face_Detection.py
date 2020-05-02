from cv2 import cv2
# utilizing already present face classifier
Face_Info = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# reading the image
my_image = cv2.imread("face.jpg")

# reading the image as greyscale 

grey_img = cv2.cvtColor(my_image,cv2.COLOR_BGR2GRAY)
#cv2.imshow('temp',grey_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows

# seach and narrow face rectangle in the images
face_box = Face_Info.detectMultiScale(grey_img,scaleFactor=1.05,minNeighbors=5)

for x,y,w,h in face_box:
    my_image = cv2.rectangle(my_image, (x,y), (x+w,y+h), (0,255,0),3)
    
'this code resizes the image and displays the result'
#resized = cv2.resize(my_image,(int(my_image.shape[1]/4),(int(my_image.shape[0]/4))))
cv2.imshow("resized", my_image)
cv2.imwrite("detected_face.jpg",my_image)
cv2.waitKey(0)
cv2.destroyAllWindows