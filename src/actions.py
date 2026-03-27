"""
actions.py  —  maps gesture labels → system actions
─────────────────────────────────────────────────────
Each action has a cooldown (seconds) to prevent repeat-firing on the same held gesture.
"""

import time
import pyautogui

pyautogui.FAILSAFE = True   # move mouse to top-left corner to abort

# Seconds between repeat triggers for the same gesture
COOLDOWNS = {
    "open_palm":   1.5,   # play/volume up  — slow so volume doesn't rocket
    "fist":        1.5,   # mute/pause
    "index_up":    0.15,  # scroll up       — fast repeat feels natural
    "two_fingers": 0.15,  # scroll down
    "pinch":       0.8,   # volume down
    "cursor_mode": 0.0,   # handled frame-by-frame in run.py
}

_last_fired: dict[str, float] = {}


def can_fire(gesture: str) -> bool:
    now = time.time()
    cooldown = COOLDOWNS.get(gesture, 1.0)
    last = _last_fired.get(gesture, 0.0)
    return (now - last) >= cooldown


def fire(gesture: str, hand_x: float | None = None, hand_y: float | None = None):
    """
    Execute the system action for the given gesture.

    hand_x, hand_y: normalized [0,1] palm position (used for cursor_mode).
    """
    if not can_fire(gesture):
        return

    _last_fired[gesture] = time.time()

    if gesture == "open_palm":
        pyautogui.press("playpause")
        # Also nudge volume up
        pyautogui.press("volumeup")

    elif gesture == "fist":
        pyautogui.press("volumemute")

    elif gesture == "index_up":
        pyautogui.scroll(3)           # positive = up

    elif gesture == "two_fingers":
        pyautogui.scroll(-3)          # negative = down

    elif gesture == "pinch":
        pyautogui.press("volumedown")

    elif gesture == "cursor_mode":
        # Map hand center to screen coordinates
        if hand_x is not None and hand_y is not None:
            sw, sh = pyautogui.size()
            # Invert x because webcam is mirrored
            screen_x = int((1 - hand_x) * sw)
            screen_y = int(hand_y * sh)
            pyautogui.moveTo(screen_x, screen_y, duration=0.05)


# Human-readable descriptions shown in the overlay UI
DESCRIPTIONS = {
    "open_palm":   "Play / Vol +",
    "fist":        "Mute / Pause",
    "index_up":    "Scroll up",
    "two_fingers": "Scroll down",
    "pinch":       "Volume down",
    "cursor_mode": "Move cursor",
}
