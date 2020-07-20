################################################################################################
#
# VISION POR COMPUTADOR
# CURSO 2018/2019
# TRABAJO 2
# FEDERICO RAFAEL GARCIA GARCIA
#
################################################################################################

################################################################################################
# IMPORTAMOS LAS LIBRERIAS NECESARIAS

import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import math

######################################
# AUTOR
######################################

print("######################################")
print("# VISION POR COMPUTADOR")
print("# CURSO 2018-2019")
print("# PRACTICA 1")
print("# FEDERICO RAFAEL GARCIA GARCIA")
print("######################################")

# Usar
# Hacer la homografia simple, no hace falta hacer difuminar ambas
# imagenes para que se vea bonito.
# No ir recalculando las homografias continuamente;
# calcularlas al principio y usarlas para generar la escena final

######################################
# FUNCIONES AUXILIARES
######################################

# Carga una imagen en RGB
def imCargarRGB(path):
    img = cv2.imread(path, 3)
    return cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)

# Carga una imagen en RGBA
def imCargarRGBA(path):
    img = cv2.imread(path, 3)
    return cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGBA)

# Pinta imagenes en una ventana, cada una con su titulo
#
#   vim: vector de imagenes
#   fils: numero de filas
#   cols: numero de columnas
#   titulos: vector de strings
#
def pintaMITitulos(vim, titulos, fils, cols):

    # Numero de imagenes
    n = len(vim)

    # En un bucle vamos mostrando cada imagen
    for i in range(0, n):
        plt.subplot(fils, cols, i+1)

        # Escondemos los ejes (simplemente por el aspecto visual)
        plt.xticks([]), plt.yticks([])

        # Mostramos la imagen
        plt.imshow(vim[i])

        # Asignamos titulo
        plt.title(titulos[i])

    # En caso de verse en una ventana (al ejecutarse desde el terminal),
    # intentar ver en pantalla completa.
    try:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
    except:
        #No hacer nada
        pass

    # Mostrar ventana
    plt.show();

    pass

#####################
# VARIABLES GLOBALES
#####################

# Variables globales usadas por las diferentes funciones

# SURF, SIFT
surf = None
sift = None

# Imagenes
yosemite1 = imCargarRGB("imagenes/yosemite1.jpg")
yosemite2 = imCargarRGB("imagenes/yosemite2.jpg")

yosemite = [None] * 3

yosemite[0] = imCargarRGBA("imagenes/yosemite1.jpg")
yosemite[1] = imCargarRGBA("imagenes/yosemite2.jpg")
yosemite[2] = imCargarRGBA("imagenes/yosemite3.jpg")

mosaico = [None] * 10

mosaico[0] = imCargarRGBA("imagenes/mosaico002.jpg")
mosaico[1] = imCargarRGBA("imagenes/mosaico003.jpg")
mosaico[2] = imCargarRGBA("imagenes/mosaico004.jpg")
mosaico[3] = imCargarRGBA("imagenes/mosaico005.jpg")
mosaico[4] = imCargarRGBA("imagenes/mosaico006.jpg")
mosaico[5] = imCargarRGBA("imagenes/mosaico007.jpg")
mosaico[6] = imCargarRGBA("imagenes/mosaico008.jpg")
mosaico[7] = imCargarRGBA("imagenes/mosaico009.jpg")
mosaico[8] = imCargarRGBA("imagenes/mosaico010.jpg")
mosaico[9] = imCargarRGBA("imagenes/mosaico011.jpg")

# Colores
colores = [
(255,   0,   0), # rojo
(  0, 200,   0), # verde
(  0,   0, 255), # azul
(255,   0, 255), # magenta
(  0, 255, 255), # cyan
(128,   0, 128), # violeta
(255, 196,   0)  # naranja
]

# Keypoints
kps_yosemite1_surf = None
kps_yosemite1_sift = None
kps_yosemite2_surf = None
kps_yosemite2_sift = None

# Keypoints estructurados por octavas y capas
kpse_yosemite1_surf = None
kpse_yosemite1_sift = None
kpse_yosemite2_surf = None
kpse_yosemite2_sift = None

# Descriptores
des_yosemite1_surf = None
des_yosemite2_surf = None
des_yosemite1_sift = None
des_yosemite2_sift = None

#####################
# EJERCICIO 1
#####################

# Dado un punto SIFT obtiene la capa, octava y escala.
# La primer octava es -1; sumamos 1 para que esta sea 0.
# La primer capa es 1; restamos 1 para que esta sea 0.
def unpackSIFTOctave(kpt):
    _octave = kpt.octave
    octave = _octave&0xFF
    layer  = (_octave>>8)&0xFF

    if octave>=128:
        octave |= -128
    if octave>=0:
        scale = float(1/(1<<octave))
    else:
        scale = float(1<<-octave)

    octave += 1
    layer -= 1
    return (octave, layer, scale)

# Dado un vector de keypoints SURF, devuelve un vector de vectores
# con los keypoints por octava
#
# kp: keypoints
# n: numero de octavas
#
def GetKeypointsPorOctavaSURF(kp):
    # Buscamos el numero de octavas
    # (el valor maximo de la variable octave)
    kp_max = kp[0].octave
    for i in range(1, len(kp)):

        if(kp[i].octave > kp_max):
            kp_max = kp[i].octave
            
    # Crear vector de vectores
    puntos_por_octava = [[] for _ in range(kp_max+1)]

    # Meter
    for point in kp:
        puntos_por_octava[point.octave].append(point)

    return puntos_por_octava

# Dado un vector de keypoints SIFT, devuelve un vector (octavas) de
# vectores (capas) de vectores (puntos)
#
# kp: keypoints
#
def GetKeypointsPorOctavaSIFT(kps):
    # Buscamos el numero de octavas
    # (el valor maximo de la variable octave)
    kp_max = -1
    for kp in kps:
        octave, layer, scale = unpackSIFTOctave(kp)
        
        if(octave > kp_max):
            kp_max = octave

    # Crear vector de vectores de vectores
    # Consiste en un vector principal que almacena vectores para cada octava.
    # Cada vector de octavas tiene 3 vectores para las 3 capas.
    # En cada vector de capas estan los puntos
    puntos_por_octava = [[[] for _ in range(3)] for _ in range(kp_max+1)]

    # Meter
    for kp in kps:
        octave, layer, scale = unpackSIFTOctave(kp)
        puntos_por_octava[octave][layer].append(kp)

    return puntos_por_octava

# Pinta los keypoints SURF con un color diferente por octava segun su sigma.
    
# imagen: imagen donde pintar los puntos
# kps: keypoints surf estructurados por octavas
# sigma: sigma inicial
# titulo: titulo de las imagenes
#
def PintarOctavasSURF(imagen, kps, sigma, titulo):

    # Copia de la original
    img = imagen.copy()

    # Sigma inicial
    s = sigma

    # Por cada octava
    for i in range(len(kps)):
        # Color segun octava
        c = colores[i]
        
        # Radio
        r = int(3*s)
            
        # Por cada punto
        for j in range(len(kps[i])):
            # Pintar circulo
            cv2.circle(img=img, center=(int(kps[i][j].pt[0]), int(kps[i][j].pt[1])), radius=r, color=c, thickness=1)
            
        # Doblar sigma
        s = s*2

    # Mostrar la imagen
    pintaMITitulos([img], [titulo], 1, 1)

    pass

# Pinta los keypoints SIFT con un color diferente por octava segun su sigma.
    
# imagen: imagen donde pintar los puntos
# kps: keypoints surf estructurados por octavas y capas
# sigma: sigma inicial
# titulo: titulo de las imagenes
#
def PintarOctavasSIFT(imagen, kps, sigma, titulo):

    # Copia de la original
    img = imagen.copy()

    # Sigma inicial
    s = sigma

    # Por cada octava
    for i in range(len(kps)):
        # Color segun octava
        c = colores[i]
        
        # Radio
        r = int(3*s)
        
        # Por cada capa
        for j in range(len(kps[i])):
            
            # Por cada punto
            for k in range(len(kps[i][j])):
                # Pintar circulo
                cv2.circle(img=img, center=(int(kps[i][j][k].pt[0]), int(kps[i][j][k].pt[1])), radius=r, color=c, thickness=1)
                
        # Doblar sigma
        s = s*2

    # Mostrar la imagen
    pintaMITitulos([img], [titulo], 1, 1)

    pass

# Muestra informacion de los puntos SURF
def PrintKeypointsSURF(kps):
    total = 0
    for i in range(len(kps)):
        print("Octava " + str(i+1) + ": " + str(len(kps[i])) + " puntos")
        total += len(kps[i])
    print("Total:", total)

# Muestra informacion de los puntos SIFT
def PrintKeypointsSIFT(kps):
    total = 0
    for i in range(len(kps)):
        print("Octava " + str(i+1) + ":")

        for j in range(len(kps[i])):
            print("\tCapa " + str(j+1) + ": " + str(len(kps[i][j])) + " puntos")
            total += len(kps[i][j])
    print("Total:", total)
    

def Ejercicio1():
    print("")
    print("#######################")
    print("#### EJERCICIO 1.A ####")
    print("#######################")

    # Para modificar las variables globales
    global surf
    global sift

    global kps_yosemite1_surf
    global kps_yosemite1_sift
    global kps_yosemite2_surf
    global kps_yosemite2_sift

    global kpse_yosemite1_surf
    global kpse_yosemite1_sift
    global kpse_yosemite2_surf
    global kpse_yosemite2_sift
    
    global des_yosemite2_surf
    global des_yosemite1_surf
    global des_yosemite2_sift
    global des_yosemite1_sift

    ############
    # SURF

    print("")
    print("---- PUNTOS SURF ----")

    # Probamos valores de Hessian Threshold para Yosemite 1, y en base a eso
    # usamos para Yosemite 2
    
    # Obtenemos los puntos SURF de Yosemite 1 - PRIMER INTENTO
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=100, nOctaves=3)
    
    kps_yosemite1_surf = surf.detect(yosemite1, None)
    imagen_puntos  = cv2.drawKeypoints(
            image=yosemite1, keypoints=kps_yosemite1_surf, outImage=None, color=colores[0],
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    
    pintaMITitulos([imagen_puntos], ["Yosemite 1 - SURF\nHT = 100 | Puntos: "+str(len(kps_yosemite1_surf))], 1, 1)
    
    # Obtenemos los puntos SURF de Yosemite 1 - SEGUNDO INTENTO
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=1000, nOctaves=3)
    
    kps_yosemite1_surf = surf.detect(yosemite1, None)
    imagen_puntos  = cv2.drawKeypoints(
            image=yosemite1, keypoints=kps_yosemite1_surf, outImage=None, color=colores[0],
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    
    pintaMITitulos([imagen_puntos], ["Yosemite 1 - SURF\nHT = 1000 | Puntos: "+str(len(kps_yosemite1_surf))], 1, 1)
    print("Puntos:", len(kps_yosemite1_surf))
    
    # Obtenemos los puntos SURF de Yosemite 1 - TERCER INTENTO
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=500, nOctaves=3)
    
    kps_yosemite1_surf = surf.detect(yosemite1, None)
    imagen_puntos  = cv2.drawKeypoints(
            image=yosemite1, keypoints=kps_yosemite1_surf, outImage=None, color=colores[0],
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    
    pintaMITitulos([imagen_puntos], ["Yosemite 1 - SURF\nHT = 500 | Puntos: "+str(len(kps_yosemite1_surf))], 1, 1)
    print("Puntos:", len(kps_yosemite1_surf))
    
    kpse_yosemite1_surf = GetKeypointsPorOctavaSURF(kps_yosemite1_surf)

     # Obtenemos los puntos SURF de Yosemite 2
    kps_yosemite2_surf = surf.detect(yosemite2, None)
    imagen_puntos  = cv2.drawKeypoints(
            image=yosemite2, keypoints=kps_yosemite2_surf, outImage=None, color=colores[0],
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    
    pintaMITitulos([imagen_puntos], ["Yosemite 2 - SURF\nHT = 500 | Puntos: "+str(len(kps_yosemite2_surf))], 1, 1)
    print("Puntos:", len(kps_yosemite2_surf))

    kpse_yosemite2_surf = GetKeypointsPorOctavaSURF(kps_yosemite2_surf)
    
    ############
    # SIFT

    print("")
    print("---- PUNTOS SIFT ----")

    # Obtenemos los puntos SIFT de Yosemite 1
    sift = cv2.xfeatures2d.SIFT_create(1000)
    
    kps_yosemite1_sift = sift.detect(yosemite1, None)
    imagen_puntos  = cv2.drawKeypoints(
            image=yosemite1, keypoints=kps_yosemite1_sift, outImage=None, color=colores[0],
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    pintaMITitulos([imagen_puntos], ["Yosemite 1 - SIFT\nPuntos: "+str(len(kps_yosemite1_sift))], 1, 1)

    kpse_yosemite1_sift = GetKeypointsPorOctavaSIFT(kps_yosemite1_sift)
    

    # Obtenemos los puntos SIFT de Yosemite 2
    kps_yosemite2_sift = sift.detect(yosemite2, None)
    imagen_puntos  = cv2.drawKeypoints(
            image=yosemite2, keypoints=kps_yosemite2_sift, outImage=None, color=colores[0],
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    pintaMITitulos([imagen_puntos], ["Yosemite 2 - SIFT\nPuntos: "+str(len(kps_yosemite2_sift))], 1, 1)

    kpse_yosemite2_sift = GetKeypointsPorOctavaSIFT(kps_yosemite2_sift)

    print("")
    print("#######################")
    print("#### EJERCICIO 1.B ####")
    print("#######################")

    # Pintamos los puntos por octavas
    PintarOctavasSURF(yosemite1, kpse_yosemite1_surf, 1.2, "Yosemite1 - SURF. Colores segun octava:\n1=Rojo, 2=Verde, 3=Azul")
    PrintKeypointsSURF(kpse_yosemite1_surf)
    
    PintarOctavasSURF(yosemite2, kpse_yosemite2_surf, 1.2, "Yosemite2 - SURF. Colores segun octava:\n1=Rojo, 2=Verde, 3=Azul")
    PrintKeypointsSURF(kpse_yosemite2_surf)
    
    PintarOctavasSIFT(yosemite1, kpse_yosemite1_sift, 1.6, "Yosemite1 - SIFT. Colores segun octava:\n1=Rojo, 2=Verde, 3=Azul\n4=Celeste, 5=Magenta, 6=Violeta, 7=Naranja")
    PrintKeypointsSIFT(kpse_yosemite1_sift)
    
    PintarOctavasSIFT(yosemite2, kpse_yosemite2_sift, 1.6, "Yosemite2 - SIFT. Colores segun octava:\n1=Rojo, 2=Verde, 3=Azul\n4=Celeste, 5=Magenta, 6=Violeta, 7=Naranja")
    PrintKeypointsSIFT(kpse_yosemite2_sift)
    
    print("")
    print("#######################")
    print("#### EJERCICIO 1C ####")
    print("#######################")

    # Extraemos los descriptores dados los keypoints del apartado anterior
    
    #SURF
    no_se_usa, des_yosemite1_surf = surf.compute(yosemite1, kps_yosemite1_surf)
    no_se_usa, des_yosemite2_surf = surf.compute(yosemite2, kps_yosemite2_surf)
    
    print("Obtenidos " + str(len(des_yosemite1_surf)) + " descriptores SURF de Yosemite1")
    print("Obtenidos " + str(len(des_yosemite2_surf)) + " descriptores SURF de Yosemite2")
   
    # SIFT
    no_se_usa, des_yosemite1_sift = sift.compute(yosemite1, kps_yosemite1_sift)
    no_se_usa, des_yosemite2_sift = sift.compute(yosemite2, kps_yosemite2_sift)
    
    print("Obtenidos " + str(len(des_yosemite1_sift)) + " descriptores SIFT de Yosemite1")
    print("Obtenidos " + str(len(des_yosemite2_sift)) + " descriptores SIFT de Yosemite2")
    
    pass

def Ejercicio2():
    

    print("")
    print("#######################")
    print("#### EJERCICIO 2.B ####")
    print("#######################")

    ###########################################################################
    # BRUTE FORCE

    # Crear Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    # Buscar los matches
    matches = bf.match(des_yosemite1_sift, des_yosemite2_sift)

    # Nos quedamos con 100 aleatorios
    matches_100 = random.sample(matches, 100);

    # Dibujar 100 matches
    imagen_matches = cv2.drawMatches(yosemite1, kps_yosemite1_sift, yosemite2, kps_yosemite2_sift, matches_100, flags=2, outImg=None)

    # Mostrar
    pintaMITitulos([imagen_matches], ["Correspondencias Yosemite - BruteForce+crossCheck"], 1, 1)

    ###########################################################################
    # LOWE

    # Crear Brute Force Matcher
    bf = cv2.BFMatcher()

    # Buscar los matches con Lowe
    matches = bf.knnMatch(des_yosemite1_sift, des_yosemite2_sift, k=2)

    # Nos quedamos con los que estan por debajo del umbral:
    # los dos puntos que mas han coincidido estan muy cerca
    # entre ellos: debe ser un buen match
    matches_buenos = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            matches_buenos.append([m])

    # Nos quedamos con 100 aleatorios
    matches_buenos_100 = random.sample(matches_buenos, 100);

    # Nos quedamos con los mejores en lugar de aleatorios
    #matches = sorted(matches, key = lambda x:x.distance)

    # Dibujar 100 matches
    imagen_matches_knn = cv2.drawMatchesKnn(yosemite1, kps_yosemite1_sift, yosemite2, kps_yosemite2_sift, matches_buenos_100, flags=2, outImg=None)

    # Mostrar
    pintaMITitulos([imagen_matches_knn], ["Correspondencias Yosemite - Lowe con k=2"], 1, 1)

    pass

# Calcula la homografia entre dos imagenes
def GetHomografia(img1, img2):
    # Obtenemos los puntos y descriptores SIFT de cada imagen
    sift = cv2.xfeatures2d.SIFT_create(1000)

    kps1, des1 = sift.detectAndCompute(img1, None)
    kps2, des2 = sift.detectAndCompute(img2, None)

    # Matcher
    bf = cv2.BFMatcher()

    # Buscar los matches con Lowe
    matches = bf.knnMatch(des1, des2, k=2)

    # Los mejores
    buenos = []

    # Nos quedamos con los buenos
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            buenos.append(m)

    # Puntos
    puntos1 = np.float32([ kps1[m.queryIdx].pt for m in buenos]).reshape(-1,1,2)
    puntos2 = np.float32([ kps2[m.trainIdx].pt for m in buenos]).reshape(-1,1,2)

    # Obtenemos la homografia
    h, mask = cv2.findHomography(puntos1, puntos2, cv2.RANSAC, 5.0)

    # La devolvemos
    return h

# Dado un vector de imagenes genera un mosaico por homografias
#
#   vi: vector de imagenes
#   n: numero de imagenes a usar
#
def GenerarMosaico(vim, n):

    # Creamos un canvas que pueda contener a todas las imagenes.
    # Hacemos que sea la suma del largo y alto de cada imagen
    canvas_w = 0
    canvas_h = 0

    for i in range (0, n):
        canvas_w += vim[i].shape[1]
        canvas_h += vim[i].shape[0]

    # La homografia original, la matriz identidad
    h = np.identity(3)

    # La imagen del medio
    nm = int(len(vim)/2)

    # Queremos colocar la imagen central en el centro del mosaico.
    # Modificamos la matriz identidad para realizar una traslacion al medio
    h[0, 2] = int((canvas_w-vim[nm].shape[1])/2)
    h[1, 2] = int((canvas_h-vim[nm].shape[0])/2)

    # Homografias
    homografias = [None] * n

    # La homografia del centro
    homografias[nm] = h.copy()
    
    # En un bucle vamos generando las homografias por la izquierda
    for i in range(nm, 0, -1):
        # Obtenemos la homografia pero "al reves":
        # nos interesa el lugar de destino en el canvas
        hom = GetHomografia(vim[i-1], vim[i])
        h = np.matmul(h, hom)
        homografias[i-1] = h

    # Reiniciamos la homografia
    h = homografias[nm].copy()

    # En un bucle vamos generando las homografias por la derecha
    for i in range(nm+1, n):
        # Obtenemos la homografia pero "al reves":
        # nos interesa el lugar de destino en el canvas
        hom = GetHomografia(vim[i], vim[i-1])
        h = np.matmul(h, hom)
        homografias[i] = h

    # Imagen donde se ira dibujando el mosaico
    canvas = np.zeros((canvas_h, canvas_w, 4), np.uint8)

    # En un bucle pintamos todas las homografias
    for i in range(0, n):
        # Pintamos en el canvas
        canvas = cv2.warpPerspective(src=vim[i], dst=canvas, M=homografias[i], dsize=(canvas_w, canvas_h), borderMode=cv2.BORDER_TRANSPARENT)

    # Recortar
    recortada = Recortar(canvas)

    # Mostrar
    pintaMITitulos([recortada], ["Mosaico N="+str(n)], 1, 1)

    pass

# Dada una imagen busca los bordes y los recorta
def Recortar(img):

    # Tamano
    w = img.shape[1]
    h = img.shape[0]

    # Posiciones de recorte
    izq = 0
    der = 0
    arr = 0
    aba = 0

    # Buscamos por columnas de izquierda a derecha
    for x in range(0, w):
        # Si algun pixel alpha no es 0 terminar
        if(np.any(img[:, x] > 0)):
            izq = x
            break

    # Buscamos por columnas de derecha a izquierda
    for x in range(0, w):
        # Si algun pixel alpha no es 0 terminar
        if(np.any(img[:, w-x-1, 3] > 0)):
            der = x
            break

    # Buscamos por filas de arriba a abajo
    for y in range(0, h):
        # Si algun pixel alpha no es 0 terminar
        if(np.any(img[y, :] > 0)):
            arr = y
            break

    # Buscamos por filas de abajo a arriba
    for y in range(0, h):
        # Si algun pixel alpha no es 0 terminar
        if(np.any(img[h-y-1, :] > 0)):
            aba = y
            break

    return img[arr:h-aba, izq:w-der]


def Ejercicio3():
    print("")
    print("#####################")
    print("#### EJERCICIO 3 ####")
    print("#####################")

    global yosemite

    GenerarMosaico(yosemite, 3)

def Ejercicio4():
    print("")
    print("#####################")
    print("#### EJERCICIO 4 ####")
    print("#####################")

    global mosaico

    GenerarMosaico(mosaico, len(mosaico))

    pass

# Ejecutamos cada ejercicio
Ejercicio1()
Ejercicio2()
Ejercicio3()
Ejercicio4()
