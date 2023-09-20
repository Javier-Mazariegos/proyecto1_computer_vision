#Javier Alejandro Mazariegos Godoy 20200223
import cvlib
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys


def float64_to_uint8(img, centered=False):
    if not centered:
        img = abs(img)

    temp = 255*(img - img.min())/(img.max() - img.min())
    
    return temp.astype(np.uint8)

def imgPreProcesamiento(img_path):
    im = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    cvlib.imgview(im)
    sobelx = cv.Sobel(im, cv.CV_64F, 1,0, ksize=5)
    sobely = cv.Sobel(im, cv.CV_64F, 0,1, ksize=5)
    sx = float64_to_uint8(sobelx)
    sy = float64_to_uint8(sobely)
    result  =  (sx+sy)
    imgbin = 255 -cv.adaptiveThreshold(result, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,127,5)
    getContornos(imgbin)

def getContornos(imgbin):
    global img
    mode = cv.RETR_TREE
    method = [cv.CHAIN_APPROX_NONE, cv.CHAIN_APPROX_SIMPLE] 
    contours, hierarchy = cv.findContours(imgbin, mode, method[1])
    getPlaca(contours)

def getPlaca(contours):
    global img
    contornos_posibles = {}
    index = -1
    color = (0,255,0) 
    thickness = 1              

    r = img.copy()

    for c in range(len(contours)): 
        mask = np.zeros_like(img)
        cv.drawContours(mask, contours, c, (255, 255, 255), thickness=cv.FILLED)
        region_of_interest = cv.bitwise_and(r, mask)
        
        #Aqui obtengo solo el rectangulo que encierra el contorno.
        x, y, w, h = cv.boundingRect(contours[c])
        img_new = region_of_interest[y:y+h, x:x+w]


        aspect_ratio = float(w)/h
        if(aspect_ratio > 1.5):
            area = cv.contourArea(contours[c])
            x,y,w,h = cv.boundingRect(contours[c])
            rect_area = w*h
            extent = float(area)/rect_area
            if(extent > 0.8):
                if(len(contornos_posibles.keys()) > 0):
                    if(list(contornos_posibles.keys())[0] < extent):
                        contornos_posibles.clear()
                        contornos_posibles[extent] = [img_new, (x,y,w,h)]
                else:
                    contornos_posibles[extent] = [img_new, (x,y,w,h)]
    if(len(contornos_posibles.keys()) > 0):
        getContornosPlaca(list(contornos_posibles.values())[0])

def getContornosPlaca(dict_contorno):
    global img
    img_dict = dict_contorno[0]
    gray = cv.cvtColor(img_dict,cv.COLOR_BGR2GRAY)
    gray =  cv.GaussianBlur(gray,(5,5),1)
    thresh_val = 100
    if(np.mean(gray) > 100):
        binarized = cv.adaptiveThreshold(gray, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,127,5)
    else:
        binarized = cv.adaptiveThreshold(gray, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,23,-2)
    kernel = np.ones((5,5),np.uint8)
    binarized = cv.morphologyEx(binarized, cv.MORPH_CLOSE, kernel)
    gray =  cv.GaussianBlur(binarized,(3,3),0)
    canny = cv.Canny(binarized, 10, 200, apertureSize=3)
    canny = cv.dilate(canny,None,iterations=1)
    x1,y1,w1,h1  = dict_contorno[1]
    mode = cv.RETR_TREE
    method = [cv.CHAIN_APPROX_NONE, cv.CHAIN_APPROX_SIMPLE] 
    contours, hierarchy = cv.findContours(canny, mode, method[1])
    imagen_final = img.copy()
    r = binarized.copy()
    for c in range(len(contours)):
    #Aqui obtengo solo el rectangulo que encierra el contorno.
        mask = np.zeros_like(binarized)
        cv.drawContours(mask, contours, c, (255, 255, 255), thickness=cv.FILLED)
        region_of_interest = cv.bitwise_and(r, mask)
        x,y,w,h = cv.boundingRect(contours[c])
        

        if(255 in region_of_interest):
            aspect_ratio = float(w)/h
            if(aspect_ratio < 1.5): 
                if(np.mean(region_of_interest) > 0.8):
                    area = cv.contourArea(contours[c])
                    rect_area = w*h
                    extent = float(area)/rect_area
                    if(extent > 0.2):   
                        cv.rectangle(imagen_final, (x1 +x,  y1 + y), (x1 +x+w, y1 +y+h), (0, 255, 0), 2)  
            
    cvlib.imgview(imagen_final)


if(len(sys.argv) >= 2):
    bandera = sys.argv[1]
    if(bandera == "--p"):
        img_path = sys.argv[2]
        img = cv.imread(img_path,cv.IMREAD_COLOR) 
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        imgPreProcesamiento(img_path)
    else:
        print("Debe de ingresar todos los parametros: proyecto1_Javier_Mazariegos.py --p ./imagenes/images108.jpg")
