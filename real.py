import cv2
import time
from deepface import DeepFace
from collections import Counter, OrderedDict
import textwrap

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

emotion_list = []
start_time = time.time()
interval_duration = 30
previous_interval_end_time = start_time

occurrences_per_interval = []
overall_dominant_emotions = []

# Define the desired order of emotions
desired_emotion_order = ['neutral', 'happy', 'fear', 'surprise', 'angry', 'sad']

while True:
    ret, frame = cap.read()
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    print(result)

    current_time = time.time()
    if current_time - previous_interval_end_time >= interval_duration:
        # Calculate overall dominant emotion for the previous interval
        emotion_counts = OrderedDict(Counter(emotion_list))
        overall_dominant_emotion = max(emotion_counts, key=emotion_counts.get)

        # Store occurrences and overall dominant emotion for the previous interval
        occurrences_per_interval.append(emotion_counts)
        overall_dominant_emotions.append(overall_dominant_emotion)

        # Reset emotion list for the new interval
        emotion_list = []
        previous_interval_end_time = current_time

    if len(result) > 0 and 'dominant_emotion' in result[0]:
        dominant_emotion = result[0]['dominant_emotion']
        emotion_list.append(dominant_emotion)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    x = result[0]['region']['x']
    y = result[0]['region']['y']
    w = result[0]['region']['w']
    h = result[0]['region']['h']

    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, dominant_emotion, (50, 50), font, 3, (0, 0, 255), 2, cv2.LINE_4)
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
elapsed_time = end_time - start_time

cap.release()
cv2.destroyAllWindows()

# Calculate overall dominant emotion for the final interval
emotion_counts = OrderedDict(Counter(emotion_list))
overall_dominant_emotion = max(emotion_counts, key=emotion_counts.get)

# Store occurrences and overall dominant emotion for the final interval
occurrences_per_interval.append(emotion_counts)
overall_dominant_emotions.append(overall_dominant_emotion)

# Print the overall time elapsed
overall_time_text = f"Overall Time: {elapsed_time} seconds"
print(overall_time_text)
print()

# Print the occurrences and overall dominant emotion for each interval
for i in range(len(occurrences_per_interval)):
    interval_occurrences = occurrences_per_interval[i]
    interval_dominant_emotion = overall_dominant_emotions[i]

    print(f"Interval {i+1} (Last 30s):")
    for emotion in desired_emotion_order:
        count = interval_occurrences.get(emotion, 0)
        print(f"{emotion} -  {count}")
    print(f"Dominant Emotion: {interval_dominant_emotion}")
    print()

# Calculate the final overall dominant emotion
all_occurrences = [occurrences for interval in occurrences_per_interval for occurrences in interval.items()]
emotion_counts_combined = Counter(dict(all_occurrences))
final_overall_dominant_emotion = max(emotion_counts_combined, key=emotion_counts_combined.get)

# Print the final overall dominant emotion
print(f"Final Overall Dominant Emotion: {final_overall_dominant_emotion}")
