import cv2
import numpy as np

def crop(path, cascade):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(cascade)
    faces = cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    image = np.array(image[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]])
    l = image.shape[0]
    x=int((l-((168/192)*l))/2)
    image = image[:,x:-x]
    y = image.shape[0]
    y = int((y-y*0.8)/2)
    x = image.shape[1]
    x = int((x-x*0.8)/2)

    image = image[y:-y,x:-x]
    image = cv2.resize(image,[168,192])

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)

    return image