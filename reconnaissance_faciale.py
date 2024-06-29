import streamlit as st
import cv2
import numpy as np

st.title('Application de détection de visages')
st.write("### Bienvenue dans l'application de détection de visages")
st.write("Téléchargez une image et ajustez les paramètres pour détecter les visages.")

# Téléchargement de l'image
fichier_uploade = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

if fichier_uploade is not None:
    bytes_fichier = np.asarray(bytearray(fichier_uploade.read()), dtype=np.uint8)
    img = cv2.imdecode(bytes_fichier, 1)
    
    # Affichage de l'image originale
    st.image(img, channels="BGR", caption='Image originale')

    # Paramètres
    min_neighbors = st.slider('minNeighbors', 1, 10, 3)
    scale_factor = st.slider('scaleFactor', 1.1, 2.0, 1.2, step=0.1)
    couleur = st.color_picker('Choisissez la couleur des rectangles', '#FF5733')

    # Convertir la couleur à un format OpenCV (BGR)
    bgr_color = tuple(int(couleur.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    # Détection des visages
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    visages = face_cascade.detectMultiScale(gris, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    # Dessiner les rectangles
    for (x, y, w, h) in visages:
        cv2.rectangle(img, (x, y), (x + w, y + h), bgr_color, 2)

    # Afficher les visages détectés
    st.image(img, channels="BGR", caption='Visages détectés')

    # Enregistrer l'image avec les visages détectés
    if st.button('Enregistrer l\'image'):
        cv2.imwrite('visages_detectes.jpg', img)
        st.write("### Image enregistrée !")
        st.image('visages_detectes.jpg', channels="BGR", caption='Image avec visages détectés')
