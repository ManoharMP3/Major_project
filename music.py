import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

# Load emotion detection model and labels
model = load_model("model.h5")
label = np.load("labels.npy")

# Initialize MediaPipe
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Streamlit UI
st.header("Emotion Based Music Player")

# Initialize session state variable
if "run" not in st.session_state:
    st.session_state["run"] = "true"

# Load saved emotion or set to empty string
try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

# Function to capture video from webcam
def capture_video():
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video from webcam.")
            break
        
        # Process the frame for emotion detection and rendering
        processed_frame = process_frame(frame)

        # Display the processed frame
        st.image(processed_frame, channels="BGR")

        # Check if the 'Recommend me songs' button is clicked
        if st.button("Recommend me songs"):
            recommend_songs()

        # Close the webcam when the user clicks the 'Close' button
        if st.button("Close"):
            cap.release()
            break

# Function to process each frame for emotion detection
def process_frame(frame):
    frm = cv2.flip(frame, 1)

    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Extract features for emotion detection
    lst = []
    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 42)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 42)

        lst = np.array(lst).reshape(1, -1)

        # Predict emotion
        pred = label[np.argmax(model.predict(lst))]
        st.write(f"Detected emotion: {pred}")

        # Save emotion to file
        np.save("emotion.npy", np.array([pred]))

    # Draw landmarks on the frame
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                           landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                           connection_drawing_spec=drawing.DrawingSpec(thickness=1))
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    return frm

# Function to recommend songs based on emotion
def recommend_songs():
    lang = st.text_input("Language")
    selected_application = st.radio("Select application:", ("YouTube", "JioSaavn", "YouTube Music"))

    if lang:
        if selected_application == "YouTube":
            webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song")
        elif selected_application == "JioSaavn":
            webbrowser.open(f"https://www.jiosaavn.com/search/song/{lang}%20{emotion}%20song")
        elif selected_application == "YouTube Music":
            webbrowser.open(f"https://music.youtube.com/search?q={lang}+{emotion}+song")

        # Reset emotion to empty string
        np.save("emotion.npy", np.array([""]))

# Start capturing video from webcam
capture_video()
