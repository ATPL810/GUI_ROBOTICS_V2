import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# --------------------------------------
# Helper function: Calculate angle
# --------------------------------------
def angle(a, b, c):
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    if mag_ba == 0 or mag_bc == 0:
        return 0
    
    cos_angle = dot / (mag_ba * mag_bc)
    cos_angle = min(1.0, max(-1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


# --------------------------------------
# Check if a finger is extended
# --------------------------------------
def is_extended(hand, tip, mid, base, threshold=160):
    ang = angle(hand.landmark[tip], hand.landmark[mid], hand.landmark[base])
    return ang > threshold


# --------------------------------------
# Custom gesture classifier
# --------------------------------------
def classify_gesture(hand):
    # Finger joints
    tips = {
        "thumb": 4,
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20
    }
    mids = {
        "thumb": 3,
        "index": 6,
        "middle": 10,
        "ring": 14,
        "pinky": 18
    }
    bases = {
        "thumb": 2,
        "index": 5,
        "middle": 9,
        "ring": 13,
        "pinky": 17
    }

    # Finger state detection
    thumb = hand.landmark[4].x < hand.landmark[3].x
    index = is_extended(hand, tips["index"], mids["index"], bases["index"])
    middle = is_extended(hand, tips["middle"], mids["middle"], bases["middle"])
    ring = is_extended(hand, tips["ring"], mids["ring"], bases["ring"])
    pinky = is_extended(hand, tips["pinky"], mids["pinky"], bases["pinky"])

    extended = sum([thumb, index, middle, ring, pinky])

    # --------------------------
    # Gesture Conditions
    # --------------------------

    # ✊ FIST
    if extended == 0:
        return "Fist"

    # ✋ OPEN PALM
    if index and middle and ring and pinky:
        return "Open Palm"

    # 👉 POINTING (Index only)
    if index and not middle and not ring and not pinky:
        return "Pointing"

    # ✌️ PEACE
    if index and middle and not ring and not pinky:
        return "Peace"

    # 🤟 ROCK SIGN (Index + Pinky)
    if index and not middle and not ring and pinky:
        return "Rock"

    # 🖖 STAR TREK / VULCAN
    # Condition: index+middle extended and ring+pinky extended BUT spacing between middle & ring is big
    dist_im = abs(hand.landmark[8].x - hand.landmark[12].x)
    dist_mr = abs(hand.landmark[12].x - hand.landmark[16].x)
    if index and middle and ring and pinky:
        if dist_mr > dist_im * 1.8:  # large gap between middle & ring
            return "Vulcan"

    # 👌 OK SIGN (thumb touching index)
    thumb_tip = hand.landmark[4]
    index_tip = hand.landmark[8]
    d = math.dist((thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y))

    if d < 0.05:  # fingers touching
        return "OK"

    return "Unknown"


# --------------------------------------
# Map gesture → Tool
# --------------------------------------
gesture_to_tool = {
    "Open Palm": "Hammer",
    "Fist": "Wrench",
    "Pointing": "Screwdriver",
    "Rock": "Plier",
    "Vulcan": "Allen Key",
    "Peace": "Measuring tape",
    "OK": "Bolt"
}


# --------------------------------------
# MAIN CODE
# --------------------------------------
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:

    while True:
        ok, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Flip horizontally for mirror effect
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        gesture = "None"
        tool = "None"

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            gesture = classify_gesture(hand)
            tool = gesture_to_tool.get(gesture, "None")

        # Display
        cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        cv2.putText(frame, f"Tool: {tool}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 3)

        cv2.imshow("Gesture → Tool Selector", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
