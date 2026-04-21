"""
ASL (American Sign Language) Hand Sign Detector with Text-to-Speech
Uses MediaPipe for hand landmark detection + rule-based gesture classification
"""

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import threading
from collections import deque, Counter

# ── TTS Engine ────────────────────────────────────────────────────────────────

class TTSEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
        self._lock = threading.Lock()
        self._busy = False

    def speak(self, text):
        if self._busy:
            return
        def _run():
            with self._lock:
                self._busy = True
                self.engine.say(text)
                self.engine.runAndWait()
                self._busy = False
        threading.Thread(target=_run, daemon=True).start()

# ── Hand Landmark Helpers ─────────────────────────────────────────────────────

def landmarks_to_array(hand_landmarks):
    """Return (21,3) numpy array of (x,y,z) in [0,1] space."""
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

def finger_states(lm, handedness="Right"):
    """
    Returns list of 5 booleans [thumb, index, middle, ring, pinky] = extended?
    Properly handles left/right hand for thumb direction.
    Uses multiple joints for more robust detection.
    """
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    mcps = [2, 5, 9, 13, 17]

    states = []

    # Thumb: direction depends on handedness
    if handedness == "Right":
        thumb_extended = lm[4][0] < lm[3][0] and lm[4][0] < lm[2][0]
    else:
        thumb_extended = lm[4][0] > lm[3][0] and lm[4][0] > lm[2][0]
    states.append(thumb_extended)

    # Other four fingers: tip.y < pip.y AND tip.y < mcp.y → finger pointing up = extended
    for i in range(1, 5):
        tip_above_pip = lm[tips[i]][1] < lm[pips[i]][1]
        tip_above_mcp = lm[tips[i]][1] < lm[mcps[i]][1]
        states.append(tip_above_pip and tip_above_mcp)

    return states  # [thumb, index, middle, ring, pinky]

def dist(a, b):
    return np.linalg.norm(a - b)

def angle_between(a, b, c):
    """Angle at point b formed by points a-b-c, in degrees."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

def curl_amount(lm, finger_tip_idx, finger_mcp_idx):
    """
    Returns how curled a finger is: distance from tip to MCP (knuckle base),
    normalised by hand size. Lower = more curled.
    """
    hand_size = dist(lm[0], lm[9])
    if hand_size == 0:
        return 0
    return dist(lm[finger_tip_idx], lm[finger_mcp_idx]) / hand_size

def finger_angle(lm, mcp, pip, tip):
    """Angle of finger curl at the PIP joint. Lower = more curled."""
    return angle_between(lm[mcp], lm[pip], lm[tip])

# ── ASL Classifier ────────────────────────────────────────────────────────────

def classify_asl(lm, handedness="Right"):
    """
    Rule-based ASL static hand-sign classifier.
    lm         : (21,3) numpy array of normalised landmarks
    handedness : "Right" or "Left"
    Returns (letter_string, confidence) or (None, 0).

    MediaPipe landmark indices (key ones used here):
        0  = wrist
        1-4  = thumb  (CMC=1, MCP=2, IP=3, tip=4)
        5-8  = index  (MCP=5, PIP=6, DIP=7, tip=8)
        9-12 = middle (MCP=9, PIP=10, DIP=11, tip=12)
        13-16= ring   (MCP=13, PIP=14, DIP=15, tip=16)
        17-20= pinky  (MCP=17, PIP=18, DIP=19, tip=20)
    """
    f = finger_states(lm, handedness)
    thumb, idx, mid, ring, pinky = f

    hand_size = dist(lm[0], lm[9])
    if hand_size == 0:
        return None, 0

    # ── Tip positions ─────────────────────────────────────────────────────────
    tip_thumb = lm[4]
    tip_idx   = lm[8]
    tip_mid   = lm[12]
    tip_ring  = lm[16]
    tip_pinky = lm[20]

    # ── Normalised distances between fingertips ───────────────────────────────
    d_thumb_idx   = dist(tip_thumb, tip_idx)   / hand_size
    d_thumb_mid   = dist(tip_thumb, tip_mid)   / hand_size
    d_thumb_ring  = dist(tip_thumb, tip_ring)  / hand_size
    d_thumb_pinky = dist(tip_thumb, tip_pinky) / hand_size
    d_idx_mid     = dist(tip_idx,   tip_mid)   / hand_size
    d_mid_ring    = dist(tip_mid,   tip_ring)  / hand_size
    d_ring_pinky  = dist(tip_ring,  tip_pinky) / hand_size
    d_idx_pinky   = dist(tip_idx,   tip_pinky) / hand_size

    # ── Curl amounts for each finger (low = tightly curled) ───────────────────
    curl_thumb = curl_amount(lm, 4,  2)
    curl_idx   = curl_amount(lm, 8,  5)
    curl_mid   = curl_amount(lm, 12, 9)
    curl_ring  = curl_amount(lm, 16, 13)
    curl_pinky = curl_amount(lm, 20, 17)

    # ── PIP joint angles (lower = more curled) ────────────────────────────────
    angle_idx   = finger_angle(lm, 5, 6, 8)
    angle_mid   = finger_angle(lm, 9, 10, 12)
    angle_ring  = finger_angle(lm, 13, 14, 16)
    angle_pinky = finger_angle(lm, 17, 18, 20)

    # ── Directional helpers ───────────────────────────────────────────────────
    # Is index finger pointing more sideways than upward?
    idx_dx = abs(lm[8][0] - lm[5][0])   # horizontal displacement tip-to-MCP
    idx_dy = abs(lm[8][1] - lm[5][1])   # vertical displacement tip-to-MCP
    idx_pointing_sideways = idx_dx > idx_dy * 1.2

    # Is index pointing downward (tip below MCP)?
    idx_pointing_down = lm[8][1] > lm[5][1] + 0.03

    # Is index pointing up (tip well above MCP)?
    idx_pointing_up = lm[8][1] < lm[5][1] - 0.05

    # Are index + middle horizontally aligned (side by side, not stacked)?
    idx_mid_horizontal = abs(lm[8][1] - lm[12][1]) < 0.05

    # Mid finger direction
    mid_pointing_sideways = abs(lm[12][0] - lm[9][0]) > abs(lm[12][1] - lm[9][1]) * 1.2

    # Thumb-index touching (for O, F, D type signs)
    thumb_idx_touching = d_thumb_idx < 0.25

    # ── Count extended fingers ────────────────────────────────────────────────
    extended_count = sum(f)

    # ── Letter rules (ordered from most-specific to least-specific) ──────────

    # ══════════════════════════════════════════════════════════════════════════
    # ── FIVE fingers extended ────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    # SPACE / 5: all five fingers extended and spread
    if thumb and idx and mid and ring and pinky:
        confidence = min(curl_idx, curl_mid, curl_ring, curl_pinky) / 1.0
        return "SPACE", min(confidence, 1.0)

    # ══════════════════════════════════════════════════════════════════════════
    # ── FOUR fingers extended ────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    # B: four fingers extended upward, thumb tucked across palm
    if not thumb and idx and mid and ring and pinky:
        # Make sure fingers are close together (not spread like W+pinky)
        confidence = 0.85 if d_idx_mid < 0.25 else 0.65
        return "B", confidence

    # F: thumb + index make circle (touching), middle+ring+pinky up
    # Check BEFORE W since F also has 3 fingers up but with thumb-index touch
    if mid and ring and pinky and not idx and thumb_idx_touching:
        confidence = max(0.5, 1.0 - d_thumb_idx * 2)
        return "F", confidence

    # W: index + middle + ring up, pinky & thumb down
    if not thumb and idx and mid and ring and not pinky:
        confidence = 0.80
        # Fingers should be spread
        if d_idx_mid > 0.15 and d_mid_ring > 0.10:
            confidence = 0.90
        return "W", confidence

    # ══════════════════════════════════════════════════════════════════════════
    # ── THREE fingers extended ───────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    # K: thumb + index + middle up, ring + pinky down
    # Thumb touches or is between index and middle
    if thumb and idx and mid and not ring and not pinky:
        # K: thumb wedged between index and middle
        thumb_between = (lm[4][1] > min(lm[8][1], lm[12][1]) and
                         lm[4][1] < max(lm[5][1], lm[9][1]))
        if thumb_between and d_thumb_idx < 0.60:
            return "K", 0.80
        # Otherwise fall through to two-finger signs below

    # ══════════════════════════════════════════════════════════════════════════
    # ── TWO fingers extended ─────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    if idx and mid and not ring and not pinky:
        # H: index + middle pointing SIDEWAYS together
        if idx_pointing_sideways and mid_pointing_sideways:
            confidence = 0.85
            if idx_mid_horizontal:
                confidence = 0.92
            return "H", confidence

        # R: index + middle CROSSED (fingers overlap horizontally)
        # The tips are very close AND the index tip crosses over the middle
        idx_crosses_mid = (
            (handedness == "Right" and lm[8][0] < lm[12][0]) or
            (handedness == "Left"  and lm[8][0] > lm[12][0])
        )
        if d_idx_mid < 0.15 and idx_crosses_mid and idx_pointing_up:
            return "R", 0.85

        # U: index + middle up, CLOSE together, pointing UP
        if not thumb and d_idx_mid < 0.20 and idx_pointing_up:
            return "U", 0.85

        # V: index + middle up, SPREAD apart (peace/victory)
        if not thumb and d_idx_mid >= 0.20 and idx_pointing_up:
            confidence = min(0.95, 0.6 + d_idx_mid)
            return "V", confidence

    # ══════════════════════════════════════════════════════════════════════════
    # ── ONE finger extended ──────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    # D: index up, all others curled into a circle with thumb
    if idx and not mid and not ring and not pinky and idx_pointing_up:
        # Thumb should touch or be near middle finger (forming a circle)
        if d_thumb_mid < 0.45:
            return "D", 0.85
        else:
            return "D", 0.70

    # G: index pointing SIDEWAYS, thumb also out (like pointing at someone)
    if thumb and idx and not mid and not ring and not pinky and idx_pointing_sideways:
        return "G", 0.85

    # P: index pointing DOWNWARD, thumb extended outward
    # (P is like K but rotated downward)
    if thumb and idx and not ring and not pinky and idx_pointing_down:
        return "P", 0.80

    # L: index pointing UP, thumb extended outward (L-shape, 90° angle)
    if thumb and idx and not mid and not ring and not pinky and idx_pointing_up:
        # Verify L-shape: thumb and index should be roughly perpendicular
        return "L", 0.85

    # I: only pinky extended
    if not thumb and not idx and not mid and not ring and pinky:
        return "I", 0.90

    # Y: thumb + pinky extended, other three curled
    if thumb and not idx and not mid and not ring and pinky:
        # Check spread between thumb and pinky
        confidence = min(0.95, 0.7 + d_thumb_pinky * 0.3)
        return "Y", confidence

    # ══════════════════════════════════════════════════════════════════════════
    # ── CLOSED FIST signs (no fingers extended) ──────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    if not idx and not mid and not ring and not pinky:

        # O: thumb and index tips form a round opening (circle shape)
        # Both thumb and index tips close together, slightly separated from others
        if thumb and d_thumb_idx < 0.30:
            # O has a clear circular opening — index curves toward thumb
            # Distinguish from A/S by checking that index tip is near thumb tip
            # but index MCP is raised (fingers curl inward, not tucked flat)
            idx_mcp_raised = lm[5][1] < lm[0][1]  # index MCP above wrist
            if idx_mcp_raised and curl_idx > 0.35:
                return "O", 0.82

        # C: curved open hand — fingers bent into C shape
        # All fingers partially curled (not fully closed, not open)
        c_curl = (0.38 < curl_idx < 0.80 and 0.38 < curl_mid < 0.80 and
                  0.38 < curl_ring < 0.80 and 0.38 < curl_pinky < 0.80)
        # In C, fingers form an arc — check that tips aren't too close together
        if c_curl and d_thumb_idx < 0.75 and d_idx_pinky > 0.20:
            return "C", 0.78

        # T: thumb inserted between index and middle
        # Thumb tip is near index PIP (pokes out between index and middle)
        t_near_idx_pip = dist(lm[4], lm[6]) / hand_size
        t_near_mid_mcp = dist(lm[4], lm[9]) / hand_size
        if thumb and t_near_idx_pip < 0.30 and t_near_mid_mcp < 0.50:
            return "T", 0.78

        # X: index finger hooked — tip curled back but DIP/PIP still raised
        # The index finger is partially extended then curled back at the DIP
        idx_dip_raised = lm[7][1] < lm[5][1]  # DIP above MCP
        idx_tip_curled = lm[8][1] > lm[7][1]  # tip below DIP (curled back)
        if not thumb and idx_dip_raised and idx_tip_curled:
            return "X", 0.80

        # E: all fingers tightly curled, tips pressing toward palm
        # Thumb tucked under/across fingers
        all_curled_tight = (curl_idx < 0.45 and curl_mid < 0.45 and
                            curl_ring < 0.45 and curl_pinky < 0.45)
        if not thumb and all_curled_tight:
            # Verify tips are near the palm (close to wrist-MCP line)
            avg_tip_to_wrist = (dist(lm[8], lm[0]) + dist(lm[12], lm[0]) +
                                dist(lm[16], lm[0]) + dist(lm[20], lm[0])) / (4 * hand_size)
            if avg_tip_to_wrist < 0.80:
                return "E", 0.78

        # M: three fingers (index+middle+ring) draped over thumb
        # Thumb tip peeks out between ring and pinky
        # All three fingers are over the thumb, so thumb tip is below/between ring-pinky
        if thumb:
            thumb_below_fingers = lm[4][1] > min(lm[8][1], lm[12][1], lm[16][1])
            thumb_near_pinky_side = dist(lm[4], lm[17]) / hand_size < 0.60
            if thumb_below_fingers and thumb_near_pinky_side and d_thumb_ring < 0.40:
                return "M", 0.72

        # N: two fingers (index+middle) draped over thumb
        # Thumb tip peeks out between middle and ring
        if thumb:
            thumb_near_ring_side = dist(lm[4], lm[13]) / hand_size < 0.50
            if thumb_near_ring_side and d_thumb_mid < 0.40 and d_thumb_ring > 0.30:
                return "N", 0.72

        # S: classic fist, thumb wrapped over front of fingers
        # Thumb lies across index/middle (thumb tip near index/middle MCPs)
        t_over_idx_mcp = dist(lm[4], lm[5]) / hand_size
        t_over_mid_mcp = dist(lm[4], lm[9]) / hand_size
        if thumb and t_over_idx_mcp < 0.45 and t_over_mid_mcp < 0.55:
            return "S", 0.75

        # A: fist with thumb on the SIDE (resting against index finger)
        # Thumb is beside the fist, not over or under
        if thumb:
            # Thumb tip is to the side of the index finger, not over it
            return "A", 0.70

        # A fallback: tight fist with thumb tucked
        return "A", 0.60

    # ══════════════════════════════════════════════════════════════════════════
    # ── Partially curled (catch-all) ─────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    # C: curved open hand — this catches cases where finger_states says
    # some fingers are "extended" but they're actually in the C-curve range
    c_curl = (0.38 < curl_idx < 0.82 and 0.38 < curl_mid < 0.82 and
              0.38 < curl_ring < 0.82 and 0.38 < curl_pinky < 0.82)
    if c_curl and d_thumb_idx < 0.75:
        return "C", 0.65

    return None, 0

# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    tts = TTSEngine()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    # ── State ─────────────────────────────────────────────────────────────────
    sentence      = ""
    word          = ""
    buffer        = deque(maxlen=20)      # recent predictions (letter, confidence)
    last_add_time = 0
    letter_hold   = 1.2    # seconds to hold sign before accepting
    sign_start    = {}     # letter -> first_seen timestamp
    speak_cooldown= 0

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Accuracy tracking
    total_frames_with_hand = 0
    stable_frames          = 0   # frames where majority vote matched with good confidence

    print("ASL Detector running. Press:")
    print("  [s] Speak sentence   [c] Clear   [q] Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result= hands.process(rgb)

        detected   = None
        confidence = 0
        vote_ratio = 0

        if result.multi_hand_landmarks:
            for hand_lm, hand_info in zip(
                result.multi_hand_landmarks,
                result.multi_handedness
            ):
                mp_draw.draw_landmarks(
                    frame, hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style(),
                )
                lm_arr    = landmarks_to_array(hand_lm)
                handedness= hand_info.classification[0].label
                letter, conf = classify_asl(lm_arr, handedness)
                buffer.append((letter, conf))

                total_frames_with_hand += 1

                # Majority vote over recent buffer
                if buffer:
                    valid = [(l, c) for l, c in buffer if l is not None]
                    if valid:
                        letter_counts = Counter(l for l, c in valid)
                        detected, votes = letter_counts.most_common(1)[0]
                        vote_ratio = votes / len(buffer)
                        # Average confidence for the winning letter
                        avg_conf = np.mean([c for l, c in valid if l == detected])
                        confidence = avg_conf * vote_ratio

                        if vote_ratio < 0.45:
                            detected = None
                            confidence = 0
                        else:
                            stable_frames += 1

        # ── Letter acceptance logic ───────────────────────────────────────────
        now = time.time()
        if detected:
            if detected not in sign_start:
                sign_start = {detected: now}  # reset for new sign
            held = now - sign_start.get(detected, now)

            # Progress bar
            progress = min(held / letter_hold, 1.0)
            bar_w = int(200 * progress)
            cv2.rectangle(frame, (w//2 - 100, h - 50), (w//2 + 100, h - 30), (50,50,50), -1)
            color = (0, 200, 100) if progress < 1.0 else (0, 255, 200)
            cv2.rectangle(frame, (w//2 - 100, h - 50), (w//2 - 100 + bar_w, h - 30), color, -1)

            if held >= letter_hold and (now - last_add_time) > letter_hold:
                if detected == "SPACE":
                    if word:
                        sentence += word + " "
                        word = ""
                elif detected == "DELETE":
                    if word:
                        word = word[:-1]
                    elif sentence:
                        sentence = sentence.rstrip()
                        if " " in sentence:
                            sentence = sentence[:sentence.rfind(" ")+1]
                        else:
                            sentence = ""
                else:
                    word += detected
                last_add_time = now
                sign_start = {}
        else:
            sign_start = {}

        # ── Calculate stability / accuracy metric ─────────────────────────────
        if total_frames_with_hand > 0:
            stability_pct = (stable_frames / total_frames_with_hand) * 100
        else:
            stability_pct = 0

        # ── Overlay UI ────────────────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 105), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        det_text = detected if detected else "---"
        conf_text = f"{confidence*100:.0f}%" if detected else ""

        # Sign detection and confidence
        cv2.putText(frame, f"Sign: {det_text}", (10, 32), font, 0.9, (0,255,180), 2)
        if conf_text:
            # Confidence color: green if high, yellow if medium, red if low
            if confidence > 0.7:
                conf_color = (0, 255, 120)
            elif confidence > 0.4:
                conf_color = (0, 255, 255)
            else:
                conf_color = (0, 100, 255)
            cv2.putText(frame, conf_text, (250, 32), font, 0.8, conf_color, 2)

        # Current word
        cv2.putText(frame, f"Word: {word}", (10, 65), font, 0.7, (255,255,100), 2)

        # Stability / accuracy indicator
        if total_frames_with_hand > 10:
            stab_text = f"Stability: {stability_pct:.0f}%"
            if stability_pct > 75:
                stab_color = (0, 255, 120)
            elif stability_pct > 50:
                stab_color = (0, 255, 255)
            else:
                stab_color = (0, 100, 255)
            cv2.putText(frame, stab_text, (w - 220, 32), font, 0.6, stab_color, 2)

        # Vote ratio bar (how consistent the buffer is)
        if detected and vote_ratio > 0:
            ratio_label = f"Match: {vote_ratio*100:.0f}%"
            cv2.putText(frame, ratio_label, (w - 220, 60), font, 0.55, (200, 200, 200), 1)
            bar_len = int(150 * vote_ratio)
            cv2.rectangle(frame, (w - 220, 67), (w - 220 + 150, 77), (60, 60, 60), -1)
            bar_color = (0, 220, 100) if vote_ratio > 0.7 else (0, 200, 255)
            cv2.rectangle(frame, (w - 220, 67), (w - 220 + bar_len, 77), bar_color, -1)

        # Bottom bar: sentence + controls
        cv2.rectangle(frame, (0, h-100), (w, h), (20,20,20), -1)
        display_sentence = (sentence + word)[-60:]
        cv2.putText(frame, display_sentence,                       (10, h-60), font, 0.65, (200,220,255), 2)
        cv2.putText(frame, "[S] Speak  [C] Clear  [Q] Quit",       (10, h-20), font, 0.5,  (160,160,160), 1)

        cv2.imshow("ASL Detector - TTS", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = ""
            word     = ""
            sign_start = {}
            # Reset accuracy tracking too
            total_frames_with_hand = 0
            stable_frames = 0
            print("Cleared.")
        elif key == ord('s'):
            full = (sentence + word).strip()
            if full and (now - speak_cooldown) > 2:
                print(f"Speaking: {full}")
                tts.speak(full)
                speak_cooldown = now
        elif key == ord(' '):
            if word:
                sentence += word + " "
                word = ""

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
