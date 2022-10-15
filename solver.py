import random
from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
import os
import src.solve as solve
from typing import Tuple
os.system('sudo apt-get install tesseract-ocr')
os.system('pip install -q pytesseract')
import pytesseract
import re
import shutil
import tensorflow as tf
from tensorflow import keras
import random


# Setup class names
with open("class_names.txt", "r") as f:  # reading them in from class_names.txt
    class_names = [names.strip() for names in f.readlines()]

model1 = tf.keras.models.load_model('model/model30.h5')
model2 = tf.keras.models.load_model('model/model15.h5')
model3 = tf.keras.models.load_model('model/model2.h5')

palabras_1 = []
# Borrar el directorio de imagenes
folder = 'output'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
folder = 'wordsPuzzle'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_words(img):
    #print(type(img))
	# str to filepath
    img = Image.open(img)
	# Display image
    # img.show()
    text = pytesseract.image_to_string(img, lang="spa+eng", config="--psm 11")
    text = text.upper()
    text = re.split('\W+', text)
    text.pop()
    #palabras_1 = text
    #print(palabras_1)
    # array to string text
    text = ' '.join(text)
    # add comma to text
    text = text.replace(" ", ",")
    return text


def getmat(listaCuadrados, filas, columnas):
    matrix = [[0 for i in range(columnas)] for j in range(filas)]
    matrixT = [[0 for i in range(columnas)] for j in range(filas)]
    listaCuadrados.sort(key=lambda y: y["altura"])
    e = 0
    for i in range(filas):
        lista2 = listaCuadrados[e:e+columnas]
        e = e+columnas
        lista2.sort(key=lambda y: y["anchura"])
        j = 0
        for lista in lista2:
            matrix[i][j] = lista["letra"]
            matrixT[i][j] = lista
            j = j + 1
    return matrix, matrixT


# Obtener las filas y las columnas de la sopa
def get_colums_and_rows(listaCuadrados):
    columnas = 0
    filas = 1
    alturaAnt = listaCuadrados[0]["altura"]
    for lista in listaCuadrados:
        if (lista["altura"] > alturaAnt + 6):
            filas = filas + 1
        alturaAnt = lista["altura"]
        if (filas == 1):
            columnas = columnas + 1
    return filas, columnas


def read_board(img, words):
    #(type(img))
	# str to filepath
    img = Image.open(img)
	# Display image
    img.show()
	# Print words
    #print("Palabras a buscar: ", palabras_1)


def solve_puzzle(img, words):
    # print(type(img))
	# str to filepath
    #print(type(words))
    #print(words)
    # img = Image.open(img)
	# Pil to opencv compatible
    pil_image = Image.open(img).convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
	# Display image
	# Print words
    img = open_cv_image
    # string to array
    words = words.split(",")
    # remove last ,
    #print(words)

    imgc = img.copy()
    imgsol = img.copy()
    imgc = cv2.cvtColor(imgc, cv2.COLOR_RGB2GRAY)
    imgc = np.invert(imgc)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
	# save the blurred image
    #cv2.imwrite("output/blur.png", blur)
	# display blurred image
    threshten = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.adaptiveThreshold(threshten, 255, 1, 1, 11, 2)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# Draw contours and save the image

    characters = np.array([
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z', 'Á', 'É', 'Í', 'Ñ', 'Ó', 'Ú', '?'])

    offset = 7
    minh = 7
    minw = 2
    i = 0

    listaCuadrados = list()
    fnt = ImageFont.truetype("fonts/Roboto-Black.ttf", 20)
    for cnt in reversed(contours):
        contCuadrados = {}
        if cv2.contourArea(cnt) > 18 and cv2.contourArea(cnt) < 1000:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if h > minh and w > minw:
                i = i + 1
                height, width = imgc.shape
                y0 = 0
                y1 = height
                x0 = 0
                x1 = width
                if height > y+h+offset:
                    y1 = y+h+offset
                if width > x+w+offset:
                    x1 = x+w+offset
                if y-offset > 0:
                    y0 = y-offset
                if x-offset > 0:
                    x0 = x-offset
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                img2 = imgc[y0:y1, x0:x1]
                img2 = cv2.resize(img2, (28, 28))
                img_array = img2.reshape(1, 28, 28, 1)
                prediction1 = np.argmax(model1.predict(img_array))
                prediction2 = np.argmax(model2.predict(img_array))
                prediction3 = np.argmax(model3.predict(img_array))
                pred = 0
                if prediction1 == prediction2 and prediction2 == prediction3:
                   pred = prediction1
                elif  prediction1 == prediction2:
                   pred = prediction1
                elif  prediction2 == prediction3:
                   pred = prediction2
                elif  prediction1 == prediction3:
                   pred = prediction3
                else:
                   pred = 32
                #print(characters[pred])
                contCuadrados["anchura"] = x
                contCuadrados["altura"] = y
                contCuadrados["centrox"] = (x + x + w)/2
                contCuadrados["centroy"] = (y + y + h)/2
                contCuadrados["letra"] = characters[pred]
                listaCuadrados.append(contCuadrados)
                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)
                draw.text(((x0+x1)/2, y0-10),
                          characters[pred], font=fnt, fill=(255, 0, 0, 0))
                img = np.array(img_pil)
    #cv2.imwrite("output/Tablero_Labels.png", img)
    filas, columnas = get_colums_and_rows(listaCuadrados)
    # print listaCuadrados
    # print(listaCuadrados)
    matrix, matrixT = getmat(listaCuadrados, filas, columnas)
    palabrasxy = []

    # print()
    # print("Palabras a buscar:")
    # for i in words:
    #    print(i)
    # print()
    # for i in range(filas):
    #     for j in range(columnas):
    #        print(matrix[i][j], end = " ")
    #     print()
    # print()
    # print()
    image_new = imgsol.copy()
    overlay = imgsol.copy()
    index = 0
    index2 = 0
    for i in words:
        #(i)
        xy_positionsvec, find = solve.find_word(matrix, i)
        if find:
            palabrasxy.append(xy_positionsvec)
            xy_positionsvec = palabrasxy[index]
            # print(len(xy_positionsvec))
            xy = xy_positionsvec[0]
            xy2 = xy_positionsvec[len(xy_positionsvec)-1]
            # print(xy["x"], " ",xy["y"])
            # print(xy2["x"], " ",xy2["y"])
            coordreal = matrixT[xy["x"]][xy["y"]]
            coordreal2 = matrixT[xy2["x"]][xy2["y"]]
            centrox = coordreal["centrox"]
            centroy = coordreal["centroy"]
            centrox2 = coordreal2["centrox"]
            centroy2 = coordreal2["centroy"]
            centrox = int(centrox)
            centroy = int(centroy)
            centrox2 = int(centrox2)
            centroy2 = int(centroy2)
            color = (random.randint(0, 255), random.randint(
                0, 255), random.randint(0, 255))
            cv2.line(overlay, (centrox, centroy), (centrox2, centroy2), color,
                     thickness=int(abs(coordreal["altura"] - coordreal["centroy"])*2))
            overlay2 = imgsol.copy()
            cv2.line(overlay2, (centrox, centroy), (centrox2, centroy2), color,
                     thickness=int(abs(coordreal["altura"] - coordreal["centroy"])*2))
            image_word = cv2.addWeighted(overlay2, 0.4, image_new, 1 - 0.4, 0)
            cv2.imwrite("wordsPuzzle/" + words[index2] + ".jpg", image_word)
            # append the image into a numpy array
            
            #print(words[index2])
    
            index += 1
        index2 += 1
    alpha = 0.4  # Transparency factor
    image_new = cv2.addWeighted(overlay, alpha, image_new, 1 - alpha, 0)
    cv2.imwrite("output/Tablero_solucion.png", image_new)
    # return the images in wordsPuzzle folder as numpy arrays
    return image_new