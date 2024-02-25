import cv2
import mediapipe as mp

# Function to detect and count thumbs up and thumbs down gestures with brackets around thumbs
def detect_and_count_gestures():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

    cap = cv2.VideoCapture(0)

    thumbs_up_count = 0
    thumbs_down_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmark coordinates for thumb
                thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]
                thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0]

                # Draw square brackets around the thumb
                thumb_size = 30
                cv2.rectangle(frame, (int(thumb_x - thumb_size), int(thumb_y - thumb_size)),
                              (int(thumb_x + thumb_size), int(thumb_y + thumb_size)), (0, 255, 0), 2)

                # Detect thumbs up or thumbs down gesture based on thumb position
                if thumb_y < frame.shape[0] // 2:  # Assuming thumbs up if thumb is above the center of the frame
                    thumbs_up_count += 1
                    break
                else:
                    thumbs_down_count += 1
                    break

        # Display the count of thumbs up and thumbs down gestures
        cv2.putText(frame, f"Thumbs Up: {thumbs_up_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Thumbs Down: {thumbs_down_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow("Hand Gestures", frame)

        # Check if 'q' key is pressed to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the function to detect and count gestures
detect_and_count_gestures()

