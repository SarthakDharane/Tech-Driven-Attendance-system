import cv2
import os
import numpy as np
from flask import Flask, request, render_template, redirect, session, url_for, flash
from datetime import date, datetime
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image
from waitress import serve
from flask_session import Session

app = Flask(__name__)
app.secret_key = 'ipsit_acadamy'  # Change this to a random secret key
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
USERNAME = 'ipsit_admin'
PASSWORD = 'password'
nimgs = 10

imgBackground = cv2.imread("2962cb31-47e0-44bd-b942-bb6959ae3faf.jpg")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))

def load_face_embedding_model():
    return load_model('static/facenet_model.h5')  # Load your pre-trained FaceNet model

def extract_embedding(img):
    img = cv2.resize(img, (160, 160))  # FaceNet expects 160x160 input
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    model = load_face_embedding_model()
    embedding = model.predict(img)
    return embedding

def train_model():
    # Logic to create and save embeddings for training data
    # This function should save embeddings to a database or a dictionary.
    embeddings = []
    labels = []
    userlist = os.listdir('static/faces')
    
    model = load_face_embedding_model()  # Load model only once
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            embedding = extract_embedding(img)
            embeddings.append(embedding.flatten())
            labels.append(user)
    
    # Save embeddings and labels to a file or database for future use
    np.save('static/embeddings.npy', embeddings)
    np.save('static/labels.npy', labels)

def load_embeddings():
    embeddings = np.load('static/embeddings.npy')
    labels = np.load('static/labels.npy')
    return embeddings, labels

def identify_face(facearray):
    embeddings, labels = load_embeddings()
    face_embedding = extract_embedding(facearray).flatten()
    distances = np.linalg.norm(embeddings - face_embedding, axis=1)
    min_index = np.argmin(distances)
    return labels[min_index] if distances[min_index] < 0.5 else "Unknown"  # Threshold for recognition

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == USERNAME and password == PASSWORD:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')

    return render_template('login.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    names, rolls, times, l = extract_attendance()
    return render_template('home1.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start', methods=['GET','POST'])
def start():
    names, rolls, times, l = extract_attendance()

    if not os.path.exists('static/facenet_model.h5'):
        return render_template('home1.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='No trained model found. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            face = frame[y:y+h, x:x+w]
            identified_person = identify_face(face)

            if identified_person != "Unknown":
                add_attendance(identified_person)
                cv2.putText(frame, identified_person, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow('Attendance', imgBackground)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home1.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            if j % 5 == 0:
                cv2.imwrite(f'{userimagefolder}/{newusername}_{i}.jpg', frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if i >= nimgs:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home1.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    serve(app, host='127.0.0.1', port='5000', threads=3)
