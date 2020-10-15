import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os

def model_loader(model_dir):
    loaded_model = load_model(model_dir)
    print("Loaded model from disk")
    return loaded_model

dir = os.path.dirname(os.path.realpath(__file__))+"/"
model=model_loader(model_dir=dir+"customCNN64.h5")
face_cascade = cv2.CascadeClassifier(dir+'haarcascade_frontalface_default.xml')
        
def classifier(image, model):
    shape=model.input_shape
    shape = shape[1:3]
    img = cv2.resize(image, shape)
    #print(shape, img.shape)
    img = img.reshape(1, shape[0], shape[1], 3)/255
    acc = model.predict(img)
    prediction = np.argmax(acc)
    classes = ["Mask", "Unmasked"]
    res = classes[prediction]
    acc = acc[0][prediction]
    acc = round(acc, 4)
    return res, acc

def face_detector(image, face_cascade):
    shape=image.shape
    gray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(faces)>0:
        result = ""
        d=20
        for i in range(0, len(faces)):
            (x, y, w, h) = faces[i]
            y = np.clip(y-d, 0, y)
            x = np.clip(x-d, 0, x)
            w = np.clip(w+2*d, w, shape[0]-w-2*d)
            h = np.clip(h+2*d, h, shape[1]-h-2*d)
            face_img = image[y:y + h, x:x + w]
            res, acc = classifier(face_img, model)
            result += f"Person {i}: {res} || Accuracy:{str(acc)[:4]} \n"
            color = (0, 255, 0)
            if res == "Unmasked":
                color=(255, 0, 0)
            
            res = f"Person: {i} {res}"
            cv2.putText(image, res, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.rectangle(image,  (x, y), (x + w, y + h), color, 2)
    else:
        result="Nothing found."
    return image, result


st.write("""
         # Face Mask Classifier Using Streamlit
         """
         )
st.write("A Simple Web App to do face mask classification.")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image_data = Image.open(file)
    st.image(image_data, use_column_width=True)
    img = image_data.convert("RGB")
    image, result=face_detector(np.asarray(img), face_cascade)
    st.write(result)
    image = Image.fromarray(image)
    st.image(image, use_column_width=True)