# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import mido
from mido import Message, MidiFile, MidiTrack  # importe modules depuis Mido
# import pyaudio

# Crée une sortie MIDI virtuelle
outport = mido.open_output('IAC Driver Bus 1')
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-yellowb", "--yellowbuffer", type=int, default=128,
                help="max yellowbuffer size")
ap.add_argument("-redb", "--redbuffer", type=int, default=128,
                help="max redbuffer size")
ap.add_argument("-whiteb", "--whitebuffer", type=int, default=128,
                help="max whitebuffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "yellow"
# ball in the HSV color space, then initialize the
# list of tracked points
yellowLower = (0, 116, 77)
yellowUpper = (23, 255, 255)
yellowpts = deque(maxlen=args["yellowbuffer"])
# define the lower and upper boundaries of the "red"
# ball in the HSV color space, then initialize the
# list of tracked points
redLower = (164, 79, 81)
redUpper = (179, 255, 255)
redpts = deque(maxlen=args["redbuffer"])
# define the lower and upper boundaries of the "white"
# ball in the HSV color space, then initialize the
# list of tracked points
whiteLower = (0, 0, 95)
whiteUpper = (26, 167, 255)
whitepts = deque(maxlen=args["whitebuffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])
# allow the camera or video file to warm up
time.sleep(2.0)
# Obtain frame size information using get() method
frame_width = int(vs.get(3))
frame_height = int(vs.get(4))
frame_size = (frame_width, frame_height)
fps = 30.0

# Spécifie le nom du fichier de sortie, le codec, la fréquence d'images (FPS), et la résolution
output_file = "SnookSnook.avi"
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # Codec pour le format AVI
# Initialise l'objet VideoWriter
out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

# définir l'intervalle de temps entre les mesures (en nombre de frames)
interval_Y = 4
interval_R = 4
interval_W = 4
# initialiser le compteur de frames depuis la dernière mesure
frame_count_Y = 0
frame_count_R = 0
frame_count_W = 0
# initialiser les coordonnées de la dernière mesure
previous_center_Y = None
previous_center_R = None
previous_center_W = None
# initialiser la liste de positions à mesurer
positions_to_measure_Y = []
positions_to_measure_R = []
positions_to_measure_W = []
# initialiser la variable qui stocke l'angle précédent
previous_angle_Y = None
previous_angle_R = None
previous_angle_W = None
# seuil de distance pour considérer que la boule est en mouvement
DISTANCE_THRESHOLD_Y = 1
DISTANCE_THRESHOLD_R = 1
DISTANCE_THRESHOLD_W = 1
# dernière position mesurée
last_position_Y = None
last_position_R = None
last_position_W = None
# Initialiser la liste de points
points_Y = []
arret_Y = []
points_R = []
arret_R = []
points_W = []
arret_W = []
# Initialiser la liste de points pour les flutes
pointsflute_Y = []
pointsflute_R = []
pointsflute_W = []

x1_Y = 0
y1_Y = 0
x2_Y = 0
y2_Y = 0
x1_R = 0
y1_R = 0
x2_R = 0
y2_R = 0
x1_W = 0
y1_W = 0
x2_W = 0
y2_W = 0

# Indique si la boule est en mouvement ou non
is_moving_Y = False
is_moving_R = False
is_moving_W = False
# variable pour réinitialiser les notes
last_note_Y = None
last_note_R = None
last_note_W = None
# variable distance
distance_Y = 0
distance_R = 0
distance_W = 0
# compteur
compteur_Y = 0
compteur_R = 0
compteur_W = 0
# compteur distance à l'arrêt
somme_Y = 0
somme_R = 0
somme_W = 0

cm_Y = 0
cm_R = 0
cm_W = 0

# billes a l'arrêt
is_stopping_Y = False
is_stopping_R = False
is_stopping_W = False
stopping_Y = []
stopping_R = []
stopping_W = []
slow_Y = False
slow_R = False
slow_W = False
debutslow_Y = 0
debutslow_R = 0
debutslow_W = 0

yellowcenterslow = None
redcenterslow = None
whitecenterslow = None

noteflute_Yslow = None
noteflute_Rslow = None
noteflute_Wslow = None

Battements = 0

action_executed = False

mid = MidiFile()          # Nous pouvons créer un nouveau fichier en appelant MidiFile
track = MidiTrack()       # sans l’argument du nom de fichier. Le fichier peut ensuite
mid.tracks.append(track)  # être enregistré à la fin en appelant la méthode save()

track.append(Message('note_on', note=1, velocity=100, time=0))

# keep looping
while True:
    # grab the current frame
    frame = vs.read()
    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break
    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = cv2.resize(frame, frame_size)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "yellow", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the yellow mask
    yellowmask = cv2.inRange(hsv, yellowLower, yellowUpper)
    yellowmask = cv2.erode(yellowmask, None, iterations=2)
    yellowmask = cv2.dilate(yellowmask, None, iterations=2)
    # find contours in the yellow mask and initialize the current
    # (x, y) center of the ball
    yellowcnts = cv2.findContours(yellowmask.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    yellowcnts = imutils.grab_contours(yellowcnts)
    center = None
    # only proceed if at least one contour was found
    if len(yellowcnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        yellowc = max(yellowcnts, key=cv2.contourArea)
        ((yellowx, yellowy), yellowradius) = cv2.minEnclosingCircle(yellowc)
        yellowM = cv2.moments(yellowc)
        yellowcenter = (int(yellowM["m10"] / yellowM["m00"]), int(yellowM["m01"] / yellowM["m00"]))
        x, y = yellowcenter
        cv2.putText(frame, f"({cm_Y}cm)", (x2_Y, y2_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 121, 175), 1)
        cv2.line(frame, (x1_Y, y1_Y), (x2_Y, y2_Y), (0, 121, 175), 2)
        # si la boule a été détectée
        if yellowcenter is not None:
            # calculer la distance par rapport à la dernière position mesurée
            if last_position_Y is not None:
                distance_Y = np.sqrt(
                    (yellowcenter[0] - last_position_Y[0]) ** 2 + (yellowcenter[1] - last_position_Y[1]) ** 2)
            else:
                distance_Y = DISTANCE_THRESHOLD_Y + 1
            # si la boule a bougé suffisamment depuis la dernière position mesurée
            if distance_Y > DISTANCE_THRESHOLD_Y:
                # Ajouter les coordonnées de la boule à la liste de points
                points_Y.append(tuple(yellowcenter))
                # ajouter les coordonnées de la boule à la liste des positions à mesurer
                positions_to_measure_Y.append(yellowcenter)
                last_position_Y = yellowcenter
                # si le nombre de frames depuis la dernière mesure est supérieur ou égal à l'intervalle défini
                if frame_count_Y >= interval_Y:
                    # récupérer les positions à mesurer
                    positions_Y = positions_to_measure_Y.copy()
                    # vider la liste des positions à mesurer
                    positions_to_measure_Y.clear()
                    # réinitialiser le compteur de frames depuis la dernière mesure
                    frame_count_Y = 0
                    # si la liste de positions contient au moins deux éléments
                    if len(positions_Y) >= 2:
                        # comparer les positions les plus éloignées
                        max_distance_Y = 0
                        max_points_Y = None
                        for i in range(len(positions_Y) - 1):
                            for j in range(i + 1, len(positions_Y)):
                                distance_Y = np.sqrt(
                                    (positions_Y[i][0] - positions_Y[j][0]) ** 2 + (
                                            positions_Y[i][1] - positions_Y[j][1]) ** 2)
                                if distance_Y > max_distance_Y:
                                    max_distance_Y = distance_Y
                                    max_points_Y = (positions_Y[i], positions_Y[j])
                        # si les positions les plus éloignées ont été trouvées
                        if max_points_Y is not None:
                            # calculer l'angle entre les deux positions
                            angle_Y = np.arctan2(max_points_Y[1][1] - max_points_Y[0][1],
                                                 max_points_Y[1][0] - max_points_Y[0][0])
                            # vérifier si l'angle a changé de 0.3 par rapport à l'angle précédent
                            if previous_angle_Y is not None and abs(angle_Y - previous_angle_Y) >= 0.3:
                                if noteflute_Yslow is not None:
                                    outport.send(mido.Message('note_off', note=noteflute_Yslow, velocity=64, channel=0))
                                    # track.append(Message('note_off', note=noteflute_Yslow, velocity=64, time=Battements))
                                    # Battements = 0
                                pointsflute_Y.append(tuple(yellowcenter))
                                compteur_Y += 1
                                # Vérifier si la liste contient plus de 2 éléments
                                if compteur_Y > 2:
                                    # Retirer le tout premier élément ajouté à la liste pointsflute
                                    pointsflute_Y.pop(0)
                                    compteur_Y -= 1
                                for i in range(len(pointsflute_Y) - 1):
                                    # Ajouter les coordonnées entre les changements d'angle
                                    distanceflute_Y = np.sqrt(
                                        (pointsflute_Y[i][0] - pointsflute_Y[i + 1][0]) ** 2 +
                                        (pointsflute_Y[i][1] - pointsflute_Y[i + 1][1]) ** 2)
                                    x1_Y, y1_Y = pointsflute_Y[i]
                                    x2_Y, y2_Y = pointsflute_Y[i + 1]

                                    # partie qui gère le midi
                                    min_v = 12
                                    max_v = 84
                                    # pour convertir le résultat et inverser
                                    midi_value_Y = 69 - (10 / 75 * distanceflute_Y) + 24
                                    centimetre_Y = distanceflute_Y / (111 / 47)
                                    noteflute_Y = int(midi_value_Y)
                                    cm_Y = int(centimetre_Y)
                                    yellowcenterslow = yellowcenter

                                    if cm_Y >= 3:
                                        if last_note_Y is not None:
                                            outport.send(mido.Message('note_off', note=last_note_Y, velocity=64, channel=0))
                                            # track.append(Message('note_off', note=last_note_Y, velocity=64, time=Battements))
                                            # Battements = 0
                                        last_note_Y = noteflute_Y
                                        # if not action_executed:
                                            # Exécutez votre action ici
                                            # track.append(Message('note_off', note=1, velocity=100, time=Battements))
                                            # cv2.putText(frame, f"YO", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 1)
                                            # Battements = 0
                                            # action_executed = True  # Marquez l'action comme exécutée
                                        # Envoyer un message MIDI (note on)
                                        outport.send(mido.Message('note_on', note=noteflute_Y, velocity=64, channel=0))
                                        # track.append(Message('note_on', note=noteflute_Y, velocity=64, time=0))
                                    is_moving_Y = True
                                    slow_Y = True

                            # mettre à jour l'angle précédent
                            previous_angle_Y = angle_Y

                # augmenter le compteur de frames depuis la dernière mesure
                frame_count_Y += 1

            if is_moving_Y:
                if 0 <= distance_Y < 1:
                    if slow_Y:
                        debutslow_Y = yellowcenterslow
                        slow_Y = False
                    stopping_Y.append(yellowcenter)
                if len(stopping_Y) >= 3:
                    if yellowcenter == stopping_Y[-3]:
                        stopping_Y = []
                        x1_Y, y1_Y = debutslow_Y
                        x2_Y, y2_Y = yellowcenter
                        distance_pixels_Y = ((x2_Y - x1_Y) ** 2 + (y2_Y - y1_Y) ** 2) ** 0.5
                        # pour convertir le résultat et inverser
                        midi_value_Yslow = 69 - (10 / 75 * distance_pixels_Y) + 24
                        centimetre_Y = distance_pixels_Y / (111 / 47)
                        noteflute_Yslow = int(midi_value_Yslow)
                        cm_Y = int(centimetre_Y)
                        if cm_Y >= 3:
                            if last_note_Y is not None:
                                outport.send(mido.Message('note_off', note=last_note_Y, velocity=64, channel=0))
                                # track.append(Message('note_off', note=last_note_Y, velocity=64, time=Battements))
                                # Battements = 0
                            pointsflute_Y = []
                            compteur_Y = 0
                            # Envoyer un message MIDI (note on)
                            outport.send(mido.Message('note_on', note=noteflute_Yslow, velocity=64, channel=0))
                            # track.append(Message('note_on', note=noteflute_Yslow, velocity=64, time=0))
                        is_moving_Y = False

    # construct a mask for the color "red", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the red mask
    redmask = cv2.inRange(hsv, redLower, redUpper)
    redmask = cv2.erode(redmask, None, iterations=2)
    redmask = cv2.dilate(redmask, None, iterations=2)
    # find contours in the red mask and initialize the current
    # (x, y) center of the ball
    redcnts = cv2.findContours(redmask.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    redcnts = imutils.grab_contours(redcnts)
    center = None
    # only proceed if at least one contour was found
    if len(redcnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        redc = max(redcnts, key=cv2.contourArea)
        ((redx, redy), redradius) = cv2.minEnclosingCircle(redc)
        redM = cv2.moments(redc)
        redcenter = (int(redM["m10"] / redM["m00"]), int(redM["m01"] / redM["m00"]))
        x, y = redcenter
        cv2.putText(frame, f"({cm_R}cm)", (x2_R, y2_R), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.line(frame, (x1_R, y1_R), (x2_R, y2_R), (0, 0, 255), 2)
        # si la boule a été détectée
        if redcenter is not None:
            # calculer la distance par rapport à la dernière position mesurée
            if last_position_R is not None:
                distance_R = np.sqrt(
                    (redcenter[0] - last_position_R[0]) ** 2 + (redcenter[1] - last_position_R[1]) ** 2)
            else:
                distance_R = DISTANCE_THRESHOLD_R + 1
            # si la boule a bougé suffisamment depuis la dernière position mesurée
            if distance_R > DISTANCE_THRESHOLD_R:
                # Ajouter les coordonnées de la boule à la liste de points
                points_R.append(tuple(redcenter))
                # ajouter les coordonnées de la boule à la liste des positions à mesurer
                positions_to_measure_R.append(redcenter)
                last_position_R = redcenter
                # si le nombre de frames depuis la dernière mesure est supérieur ou égal à l'intervalle défini
                if frame_count_R >= interval_R:
                    # récupérer les positions à mesurer
                    positions_R = positions_to_measure_R.copy()
                    # vider la liste des positions à mesurer
                    positions_to_measure_R.clear()
                    # réinitialiser le compteur de frames depuis la dernière mesure
                    frame_count_R = 0
                    # si la liste de positions contient au moins deux éléments
                    if len(positions_R) >= 2:
                        # comparer les positions les plus éloignées
                        max_distance_R = 0
                        max_points_R = None
                        for i in range(len(positions_R) - 1):
                            for j in range(i + 1, len(positions_R)):
                                distance_R = np.sqrt(
                                    (positions_R[i][0] - positions_R[j][0]) ** 2 + (
                                            positions_R[i][1] - positions_R[j][1]) ** 2)
                                if distance_R > max_distance_R:
                                    max_distance_R = distance_R
                                    max_points_R = (positions_R[i], positions_R[j])
                        # si les positions les plus éloignées ont été trouvées
                        if max_points_R is not None:
                            # calculer l'angle entre les deux positions
                            angle_R = np.arctan2(max_points_R[1][1] - max_points_R[0][1],
                                                 max_points_R[1][0] - max_points_R[0][0])
                            # vérifier si l'angle a changé de 0.3 par rapport à l'angle précédent
                            if previous_angle_R is not None and abs(angle_R - previous_angle_R) >= 0.5:
                                if noteflute_Rslow is not None:
                                    outport.send(mido.Message('note_off', note=noteflute_Rslow, velocity=64, channel=1))
                                    # track.append(Message('note_off', note=noteflute_Rslow, velocity=64, time=Battements))
                                    # Battements = 0
                                pointsflute_R.append(tuple(redcenter))
                                compteur_R += 1
                                # Vérifier si la liste contient plus de 2 éléments
                                if compteur_R > 2:
                                    # Retirer le tout premier élément ajouté à la liste pointsflute
                                    pointsflute_R.pop(0)
                                    compteur_R -= 1
                                for i in range(len(pointsflute_R) - 1):
                                    # Ajouter les coordonnées entre les changements d'angle
                                    distanceflute_R = np.sqrt(
                                        (pointsflute_R[i][0] - pointsflute_R[i + 1][0]) ** 2 +
                                        (pointsflute_R[i][1] - pointsflute_R[i + 1][1]) ** 2)
                                    x1_R, y1_R = pointsflute_R[i]
                                    x2_R, y2_R = pointsflute_R[i + 1]

                                    # partie qui gère le midi
                                    min_v = 12
                                    max_v = 84
                                    # pour convertir le résultat et inverser
                                    midi_value_R = 69 - (10 / 75 * distanceflute_R) + 24
                                    centimetre_R = distanceflute_R / (111 / 47)
                                    noteflute_R = int(midi_value_R)
                                    cm_R = int(centimetre_R)
                                    redcenterslow = redcenter

                                    if cm_R >= 3:
                                        if last_note_R is not None:
                                            outport.send(mido.Message('note_off', note=last_note_R, velocity=64, channel=1))
                                            # track.append(Message('note_off', note=last_note_R, velocity=64, time=Battements))
                                            # Battements = 0
                                        last_note_R = noteflute_R
                                        # if not action_executed:
                                            # Exécutez votre action ici
                                            # track.append(Message('note_off', note=1, velocity=100, time=Battements))
                                            # cv2.putText(frame, f"YO", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 1)
                                            # Battements = 0
                                            # action_executed = True  # Marquez l'action comme exécutée
                                        # Envoyer un message MIDI (note on)
                                        outport.send(mido.Message('note_on', note=noteflute_R, velocity=64, channel=1))
                                        # track.append(Message('note_on', note=noteflute_R, velocity=64, time=0))

                                    is_moving_R = True
                                    slow_R = True

                            # mettre à jour l'angle précédent
                            previous_angle_R = angle_R

                # augmenter le compteur de frames depuis la dernière mesure
                frame_count_R += 1

            if is_moving_R:
                if 0 <= distance_R < 1:
                    if slow_R:
                        debutslow_R = redcenterslow
                        slow_R = False
                    stopping_R.append(redcenter)
                if len(stopping_R) >= 3:
                    if redcenter == stopping_R[-3]:
                        stopping_R = []

                        x1_R, y1_R = debutslow_R
                        x2_R, y2_R = redcenter
                        distance_pixels_R = ((x2_R - x1_R) ** 2 + (y2_R - y1_R) ** 2) ** 0.5
                        # pour convertir le résultat et inverser
                        midi_value_Rslow = 69 - (10 / 75 * distance_pixels_R) + 24
                        centimetre_R = distance_pixels_R / (111 / 47)
                        noteflute_Rslow = int(midi_value_Rslow)
                        cm_R = int(centimetre_R)

                        if cm_R >= 3:
                            if last_note_R is not None:
                                outport.send(mido.Message('note_off', note=last_note_R, velocity=64, channel=1))
                                # track.append(Message('note_off', note=last_note_R, velocity=64, time=Battements))
                                # Battements = 0
                            pointsflute_R = []
                            compteur_R = 0
                            # Envoyer un message MIDI (note on)
                            outport.send(mido.Message('note_on', note=noteflute_Rslow, velocity=64, channel=1))
                            # track.append(Message('note_on', note=noteflute_Rslow, velocity=64, time=0))
                        is_moving_R = False

    # construct a mask for the color "white", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the white mask
    whitemask = cv2.inRange(hsv, whiteLower, whiteUpper)
    whitemask = cv2.erode(whitemask, None, iterations=2)
    whitemask = cv2.dilate(whitemask, None, iterations=2)
    # find contours in the white mask and initialize the current
    # (x, y) center of the ball
    whitecnts = cv2.findContours(whitemask.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    whitecnts = imutils.grab_contours(whitecnts)
    center = None
    # only proceed if at least one contour was found
    if len(whitecnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        whitec = max(whitecnts, key=cv2.contourArea)
        ((whitex, whitey), whiteradius) = cv2.minEnclosingCircle(whitec)
        whiteM = cv2.moments(whitec)
        whitecenter = (int(whiteM["m10"] / whiteM["m00"]), int(whiteM["m01"] / whiteM["m00"]))
        x, y = whitecenter
        cv2.putText(frame, f"({cm_W}cm)", (x2_W, y2_W), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.line(frame, (x1_W, y1_W), (x2_W, y2_W), (255, 255, 255), 2)
        # si la boule a été détectée
        if whitecenter is not None:
            # calculer la distance par rapport à la dernière position mesurée
            if last_position_W is not None:
                distance_W = np.sqrt(
                    (whitecenter[0] - last_position_W[0]) ** 2 + (whitecenter[1] - last_position_W[1]) ** 2)
            else:
                distance_W = DISTANCE_THRESHOLD_W + 1
            # si la boule a bougé suffisamment depuis la dernière position mesurée
            if distance_W > DISTANCE_THRESHOLD_W:
                # Ajouter les coordonnées de la boule à la liste de points
                points_W.append(tuple(whitecenter))
                # ajouter les coordonnées de la boule à la liste des positions à mesurer
                positions_to_measure_W.append(whitecenter)
                last_position_W = whitecenter
                # si le nombre de frames depuis la dernière mesure est supérieur ou égal à l'intervalle défini
                if frame_count_W >= interval_W:
                    # récupérer les positions à mesurer
                    positions_W = positions_to_measure_W.copy()
                    # vider la liste des positions à mesurer
                    positions_to_measure_W.clear()
                    # réinitialiser le compteur de frames depuis la dernière mesure
                    frame_count_W = 0
                    # si la liste de positions contient au moins deux éléments
                    if len(positions_W) >= 2:
                        # comparer les positions les plus éloignées
                        max_distance_W = 0
                        max_points_W = None
                        for i in range(len(positions_W) - 1):
                            for j in range(i + 1, len(positions_W)):
                                distance_W = np.sqrt(
                                    (positions_W[i][0] - positions_W[j][0]) ** 2 + (
                                            positions_W[i][1] - positions_W[j][1]) ** 2)
                                if distance_W > max_distance_W:
                                    max_distance_W = distance_W
                                    max_points_W = (positions_W[i], positions_W[j])
                        # si les positions les plus éloignées ont été trouvées
                        if max_points_W is not None:
                            # calculer l'angle entre les deux positions
                            angle_W = np.arctan2(max_points_W[1][1] - max_points_W[0][1],
                                                 max_points_W[1][0] - max_points_W[0][0])
                            # vérifier si l'angle a changé de 0.3 par rapport à l'angle précédent
                            if previous_angle_W is not None and abs(angle_W - previous_angle_W) >= 0.5:
                                if noteflute_Wslow is not None:
                                    outport.send(mido.Message('note_off', note=noteflute_Wslow, velocity=64, channel=2))
                                    track.append(Message('note_off', note=noteflute_Wslow, velocity=64, time=Battements))
                                    Battements = 0
                                pointsflute_W.append(tuple(whitecenter))
                                compteur_W += 1
                                # Vérifier si la liste contient plus de 2 éléments
                                if compteur_W > 2:
                                    # Retirer le tout premier élément ajouté à la liste pointsflute
                                    pointsflute_W.pop(0)
                                    compteur_W -= 1
                                for i in range(len(pointsflute_W) - 1):
                                    # Ajouter les coordonnées entre les changements d'angle
                                    distanceflute_W = np.sqrt(
                                        (pointsflute_W[i][0] - pointsflute_W[i + 1][0]) ** 2 +
                                        (pointsflute_W[i][1] - pointsflute_W[i + 1][1]) ** 2)
                                    x1_W, y1_W = pointsflute_W[i]
                                    x2_W, y2_W = pointsflute_W[i + 1]

                                    # partie qui gère le midi
                                    min_v = 12
                                    max_v = 84
                                    # pour convertir le résultat et inverser
                                    midi_value_W = 69 - (10 / 75 * distanceflute_W) + 24
                                    centimetre_W = distanceflute_W / (111 / 47)
                                    noteflute_W = int(midi_value_W)
                                    cm_W = int(centimetre_W)
                                    whitecenterslow = whitecenter

                                    if cm_W >= 3:
                                        if last_note_W is not None:
                                            outport.send(mido.Message('note_off', note=last_note_W, velocity=64, channel=2))
                                            track.append(Message('note_off', note=last_note_W, velocity=64, time=Battements))
                                            Battements = 0
                                        last_note_W = noteflute_W
                                        if not action_executed:
                                            # Exécutez votre action ici
                                            track.append(Message('note_off', note=1, velocity=100, time=Battements))
                                            cv2.putText(frame, f"YO", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 1)
                                            Battements = 0
                                            action_executed = True  # Marquez l'action comme exécutée
                                        # Envoyer un message MIDI (note on)
                                        outport.send(mido.Message('note_on', note=noteflute_W, velocity=64, channel=2))
                                        track.append(Message('note_on', note=noteflute_W, velocity=64, time=0))
                                    is_moving_W = True
                                    slow_W = True

                            # mettre à jour l'angle précédent
                            previous_angle_W = angle_W

                # augmenter le compteur de frames depuis la dernière mesure
                frame_count_W += 1

            if is_moving_W:
                if 0 <= distance_W < 1:
                    if slow_W:
                        debutslow_W = whitecenterslow
                        slow_W = False
                    stopping_W.append(whitecenter)
                if len(stopping_W) >= 3:
                    if whitecenter == stopping_W[-3]:
                        stopping_W = []
                        x1_W, y1_W = debutslow_W
                        x2_W, y2_W = whitecenter
                        distance_pixels_W = ((x2_W - x1_W) ** 2 + (y2_W - y1_W) ** 2) ** 0.5
                        # pour convertir le résultat et inverser
                        midi_value_Wslow = 69 - (10 / 75 * distance_pixels_W) + 24
                        centimetre_W = distance_pixels_W / (111 / 47)
                        noteflute_Wslow = int(midi_value_Wslow)
                        cm_W = int(centimetre_W)

                        if cm_W >= 3:
                            if last_note_W is not None:
                                outport.send(mido.Message('note_off', note=last_note_W, velocity=64, channel=2))
                                track.append(Message('note_off', note=last_note_W, velocity=64, time=Battements))
                                Battements = 0
                            pointsflute_W = []
                            compteur_W = 0
                            # Envoyer un message MIDI (note on)
                            outport.send(mido.Message('note_on', note=noteflute_Wslow, velocity=64, channel=2))
                            track.append(Message('note_on', note=noteflute_Wslow, velocity=64, time=0))
                        is_moving_W = False

    Battements += 1

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Enregistrez la trame dans la vidéo
    out.write(frame)
    # enregistre le tout dans ce fichier Midi
    mid.save('MIDO_Write-Midi-File.mid')  # enregistre le tout dans ce fichier Midi

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        # outport.send(mido.Message('note_off', note=76, velocity=64, channel=4))
        allnotes = 1
        while allnotes < 128:
            outport.send(mido.Message('note_off', note=allnotes, velocity=64, channel=0))
            outport.send(mido.Message('note_off', note=allnotes, velocity=64, channel=1))
            outport.send(mido.Message('note_off', note=allnotes, velocity=64, channel=2))
            allnotes += 1
        break
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()
# otherwise, release the camera
else:
    vs.release()
    # stream.stop_stream()
    # stream.close()
    # audio_stream.terminate()
# close all windows
# outport.send(mido.Message('note_off', note=76, velocity=64, channel=4))
cv2.destroyAllWindows()
