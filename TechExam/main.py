import face_recognition
import cv2
import numpy as np
import os
from datetime import date
import xlrd
from gtts import gTTS
from xlutils.copy import copy as xl_copy
import pyfirmata
import pygame
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

def send_email(filename, recipient_email):
    # Paramètres du serveur SMTP
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = 'ouiameb73@gmail.com'
    smtp_password = 'evxfjggmankyetuz'

    # Adresse e-mail de l'expéditeur
    sender_email = 'ouiameb73@gmail.com'

    # Créer l'objet MIMEMultipart pour le message
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = 'TechExam - Fichier Excel des présences'

    # Attacher le fichier Excel
    with open(filename, 'rb') as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename= {filename}')
        message.attach(part)

    # Envoyer l'e-mail via le serveur SMTP
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(sender_email, recipient_email, message.as_string())




board = pyfirmata.Arduino('COM3')
board.digital[5].mode = pyfirmata.SERVO
red=6
green=7
trigger_pin=6
echo_pin=7
button_pin = 4
buzzer_pin = 5

it = pyfirmata.util.Iterator(board)
it.start()
time.sleep(1)


def rotateServo(angle):
    board.digital[5].write(angle)
    time.sleep(0.015)

def msg( text ):
    if text:
        board.send_sysex( pyfirmata.STRING_DATA, pyfirmata.util.str_to_two_byte_iter( text ) )

def light(pin,n):
    board.digital[pin].write(n)




def generate_audio(text, filename):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)

def play_audio(file):
    pygame.init()
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()

def analog_data_received(pin):
    if pin == echo_pin:
        dist_cm=board.digital[pin].read()
        print("Distance:", dist_cm, "cm")

def activate_buzzer():
    board.digital[buzzer_pin].write(1)  # Turn on the buzzer
    time.sleep(4)  # Buzzer duration
    board.digital[buzzer_pin].write(0)  # Turn off the buzzer



# Register the event handler for analog data
board.add_cmd_handler(pyfirmata.pyfirmata.ANALOG_MESSAGE, analog_data_received)

# Keep the program running to receive data


# Get the current folder path
CurrentFolder = os.getcwd()

# Define the paths to the sample images
image_path1 = os.path.join(CurrentFolder, 'Ouiame.png')
image_path2 = os.path.join(CurrentFolder, 'Amal.png')
image_path3 = os.path.join(CurrentFolder, 'Pr_Kobbane.png')
#image_path4 = os.path.join(CurrentFolder, 'Khaoula.png')

# Load the sample images and encode the faces
person1_image = face_recognition.load_image_file(image_path1)
person1_face_encoding = face_recognition.face_encodings(person1_image)[0]

person2_image = face_recognition.load_image_file(image_path2)
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]

person3_image = face_recognition.load_image_file(image_path3)
person3_face_encoding = face_recognition.face_encodings(person3_image)[0]

#person4_image = face_recognition.load_image_file(image_path4)
#person4_face_encoding = face_recognition.face_encodings(person4_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    person1_face_encoding,
    person2_face_encoding,
    person3_face_encoding,
    #person4_face_encoding
]
known_face_names = [
    "Ouiame",
    "Amal",
    "Pr_Kobbane",
     #"Khaoula"
]

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Open the attendance Excel file
rb = xlrd.open_workbook('attendance_excel.xls', formatting_info=True)
wb = xl_copy(rb)

# Get the input for the current subject lecture name
# inp = input('Please give the exam subject name: ')

# Define the additional information
student_info = {
    "Ouiame": {
        "prénom": "BAHOU",
        "numéro de table": 24,
        "numéro de salle": 3,
        "prof surveillant": "Abdellatif Kobbane",
        "matière d'examen": "Réseaux Informatiques",
        "durée d'examen": "2 heures"
    },
    "Amal": {
        "prénom": "ABERGHOU",
        "numéro de table": 25,
        "numéro de salle": 3,
        "prof surveillant": "Abdellatif Kobbane",
        "matière d'examen": "Réseaux Informatiques",
        "durée d'examen": "2 heures"
    },
    "Khaoula": {
        "prénom": "MEJHOUDI",
        "numéro de table": 26,
        "numéro de salle": 3,
        "prof surveillant": "Abdellatif Kobbane",
        "matière d'examen": "Réseaux Informatiques",
        "durée d'examen": "2 heures"
    },
}
#number of students to take attendance before sending the mail
num_total=2
# Get the current date as the sheet name
sheet_name = str(date.today())

# Add a new sheet to the workbook with the current date as the name
sheet1 = wb.add_sheet(sheet_name)

# Write column titles
sheet1.write(0, 0, 'Nom')
sheet1.write(0, 1, 'Prénom')
sheet1.write(0, 2, 'Numéro de table')
sheet1.write(0, 3, 'Numéro de salle')
sheet1.write(0, 4, 'Prof surveillant')
sheet1.write(0, 5, 'Matière d\'examen')
sheet1.write(0, 6, 'Durée d\'examen')

row = 1
already_attendance_taken = set()
file1 = "welcome_audio.mp3"
file2 = "No.mp3"
file3="end.mp3"


while True:

    rotateServo(150);

    light(red,1)
    light(green,0)
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Inconnu"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                info = student_info[name]
                if (name != "Pr_Kobbane"):
                    prenom=info["prénom"]
                    num_t=str(info["numéro de table"])
                    lcd_msg=name+" "+prenom
                    light(red, 0)
                    light(green, 1)
                    play_audio(file1)
                    msg(lcd_msg)
                    msg("Num: " + num_t)
                    time.sleep(5);

                    rotateServo(0)
                    time.sleep(10)
                    rotateServo(150)
                    msg(" Looking for")
                    msg(" face ... ")
                    time.sleep(3)
                if (name not in already_attendance_taken) and (name != "Inconnu"):
                    # Write the student information to the Excel sheet
                    sheet1.write(row, 0, name)
                    sheet1.write(row, 1, info["prénom"])
                    sheet1.write(row, 2, info["numéro de table"])
                    sheet1.write(row, 3, info["numéro de salle"])
                    sheet1.write(row, 4, info["prof surveillant"])
                    sheet1.write(row, 5, info["matière d'examen"])
                    sheet1.write(row, 6, info["durée d'examen"])
                    row += 1
                    wb.save('attendance_excel.xls')
                    already_attendance_taken.add(name)
                    print("Attendance taken for", name)
                if(name=="Pr_Kobbane"):
                    print("Professor recognized")
                    msg("Professor ")
                    msg("A.Kobbane ")
                    light(red, 0)
                    light(green, 1)
                    play_audio(file1)


                    rotateServo(0)
                    time.sleep(10)
                    rotateServo(150)
                    msg(" Looking for")
                    msg(" face ... ")
                    time.sleep(5)


            else:
                play_audio(file2)
                msg("Face")
                msg("Not recognized")
                time.sleep(4)
                msg(" Looking for")
                msg(" face ... ")
                print("Next Student")

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Data saved")
        break

video_capture.release()
num_taken = len(already_attendance_taken)
print(num_taken)

# Release the webcam
if num_taken==num_total:
    # Utilisation de la fonction send_email pour envoyer le fichier Excel au professeur
    filename = 'attendance_excel.xls'
    professor_email = 'wiamebahou@gmail.com'
    play_audio(file3)

    time.sleep(6)
    send_email(filename, professor_email)

cv2.destroyAllWindows()
