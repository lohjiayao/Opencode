import streamlit as st
import cv2
import numpy as np
import imutils
import pandas as pd
import time
import threading
import queue

# --- 1. GLOBAL SYSTEM ARCHITECTURE CONFIG ---
st.set_page_config(page_title="AI Sorter: Calibrated Vision", layout="wide")

# --- DEFAULT CONFIG ---
CAMERA_INDEX = 0    
CAMERA_BACKEND = cv2.CAP_DSHOW
SMOOTHING_ALPHA = 0.25          # Smoothing factor for tracking stability
ANGLE_DEADBAND = 1.5            # Ignore angular jitter smaller than 1.5 degrees

class IndustrialVisionSystem:
    def __init__(self):
        self.running = False
        self.cap = None
        self.thread = None

        # Thread-safe data channels
        self.product_db = {}
        self.product_db_lock = threading.Lock()

        # Calibration parameters (Thread-Safe)
        self.pixels_per_mm = 1.0  # Default 1:1 fallback
        self.cal_lock = threading.Lock()

        # Command & Result queues
        self._cmd_queue = queue.Queue(maxsize=1)
        self._result_queue = queue.Queue(maxsize=1)

        self.system_status = "Initializing..."
        self._status_lock = threading.Lock()

        # Temporal smoothing memory
        self.sm_w = None
        self.sm_h = None
        self.sm_angle = None

    def set_calibration(self, target_mm: float, raw_pixels: float):
        with self.cal_lock:
            self.pixels_per_mm = raw_pixels / target_mm
            
    def get_pixels_per_mm(self) -> float:
        with self.cal_lock:
            return self.pixels_per_mm
        

    def _set_status(self, msg: str):
        with self._status_lock:
            self.system_status = msg

    def get_status(self) -> str:
        with self._status_lock:
            return self.system_status

    def start_hardware(self):
        if self.running:
            return
        self.running = True
        self.cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # === PASTE THE NEW FOCUS LOGIC HERE ===
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)    # Turns off automatic camera guessing
        time.sleep(0.1)                            # Gives hardware a brief moment to switch modes
        self.cap.set(cv2.CAP_PROP_FOCUS, 1)       # Adjust this number (0 to 50) until it is sharp!
        # ======================================

        self.thread = threading.Thread(target=self._vision_loop, daemon=True, name="VisionWorker")
        self.thread.start()

    def stop_hardware(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        self._set_status("Offline")

    def _vision_loop(self):
        self._set_status("Online")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        while self.running:
            time.sleep(0.01)

            if not self.cap or not self.cap.isOpened():
                self._set_status("Error: Camera disconnected")
                break

            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                processing_frame = imutils.resize(frame, width=500)
                gray = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edged = cv2.Canny(blurred, 60, 160)
                
                edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
                edged = cv2.dilate(edged, None, iterations=1)

                cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                best_contour = None
                max_area = 0
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area > 1000 and area > max_area:
                        max_area = area
                        best_contour = c

                pending_cmd = None
                try:
                    pending_cmd = self._cmd_queue.get_nowait()
                except queue.Empty:
                    pass

                if best_contour is not None:
                    rect = cv2.minAreaRect(best_contour)
                    (x, y), (w, h), raw_angle = rect
                    
                    raw_w = max(w, h)
                    raw_h = min(w, h)
                    raw_angle = raw_angle + 90 if w < h else raw_angle

                    # --- Anti-Jitter Smoothing Engine ---
                    if self.sm_w is None:
                        self.sm_w = raw_w
                        self.sm_h = raw_h
                        self.sm_angle = raw_angle
                    else:
                        self.sm_w = (SMOOTHING_ALPHA * raw_w) + ((1.0 - SMOOTHING_ALPHA) * self.sm_w)
                        self.sm_h = (SMOOTHING_ALPHA * raw_h) + ((1.0 - SMOOTHING_ALPHA) * self.sm_h)
                        
                        angle_delta = raw_angle - self.sm_angle
                        if angle_delta > 180: angle_delta -= 360
                        elif angle_delta < -180: angle_delta += 360
                        
                        if abs(angle_delta) > ANGLE_DEADBAND:
                            self.sm_angle = self.sm_angle + (SMOOTHING_ALPHA * angle_delta)

                    # --- Convert Raw Pixels to Millimeters ---
                    current_scale = self.get_pixels_per_mm()
                    mm_w = round(self.sm_w / current_scale, 2)
                    mm_h = round(self.sm_h / current_scale, 2)
                    detected_angle = self.sm_angle
                    aspect_ratio = round(mm_w / mm_h, 2)

                    # --- Process Save Command ---
                    if pending_cmd is not None:
                        if pending_cmd["type"] == "TRAIN":
                            with self.product_db_lock:
                                self.product_db[pending_cmd["name"]] = {
                                    "expected_width_mm": mm_w,
                                    "expected_height_mm": mm_h,
                                    "aspect_ratio": aspect_ratio,
                                    "baseline_angle": detected_angle,
                                    "target_slot": pending_cmd["slot"],
                                    "tolerance": pending_cmd["tolerance"]
                                }
                            self._result_queue.put_nowait(("SUCCESS", "TRAIN"))
                        elif pending_cmd["type"] == "CALIBRATE":
                            # Set calibration ratio based on current stabilized pixel width
                            self.set_calibration(pending_cmd["target_mm"], self.sm_w)
                            self._result_queue.put_nowait(("SUCCESS", "CALIBRATE"))
                        pending_cmd = None

                    # --- Real-World Classification Logic ---
                    matched_product = "Unknown"
                    final_corrected_angle = 0.0
                    assigned_slot = 0
                    best_confidence = float("inf")

                    with self.product_db_lock:
                        db_snapshot = dict(self.product_db)

                    for name, profile in db_snapshot.items():
                        # Compare metrics in true millimeters, not erratic pixels!
                        w_diff = abs(mm_w - profile["expected_width_mm"]) / profile["expected_width_mm"]
                        h_diff = abs(mm_h - profile["expected_height_mm"]) / profile["expected_height_mm"]
                        allowed_tol = profile["tolerance"]

                        if w_diff < allowed_tol and h_diff < allowed_tol:
                            confidence = w_diff + h_diff
                            if confidence < best_confidence:
                                best_confidence = confidence
                                matched_product = name
                                assigned_slot = profile["target_slot"]
                                
                                rot_delta = detected_angle - profile["baseline_angle"]
                                if rot_delta > 180: rot_delta -= 360
                                elif rot_delta < -180: rot_delta += 360
                                final_corrected_angle = rot_delta

                    # Double check for sizing ambiguity
                    ambiguous_count = 0
                    for profile in db_snapshot.values():
                        w_d = abs(mm_w - profile["expected_width_mm"]) / profile["expected_width_mm"]
                        h_d = abs(mm_h - profile["expected_height_mm"]) / profile["expected_height_mm"]
                        if w_d < profile["tolerance"] and h_d < profile["tolerance"]:
                            ambiguous_count += 1
                    if ambiguous_count > 1:
                        matched_product = "Ambiguous"

                    # --- Render Calibrated Overlay ---
                    if w < h:
                        draw_rect = ((x, y), (self.sm_h, self.sm_w), detected_angle - 90)
                    else:
                        draw_rect = ((x, y), (self.sm_w, self.sm_h), detected_angle)
                    box = cv2.boxPoints(draw_rect).astype(np.int32)

                    color = (0, 255, 0) if matched_product not in ("Unknown", "Ambiguous") else (0, 0, 255)
                    cv2.drawContours(processing_frame, [box], 0, color, 2)
                    
                    # Show live millimeter readings on screen
                    metrics_label = f"{mm_w}mm x {mm_h}mm | {int(final_corrected_angle)}deg"
                    cv2.putText(processing_frame, metrics_label, (int(box[0][0]), int(box[0][1]) - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    
                    class_label = f"ID: {matched_product} -> Bin {assigned_slot}"
                    cv2.putText(processing_frame, class_label, (int(box[0][0]), int(box[0][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                else:
                    self.sm_w, self.sm_h, self.sm_angle = None, None, None
                    if pending_cmd is not None:
                        self._result_queue.put_nowait(("ERROR_EMPTY", "NONE"))

                cv2.imshow("Industrial Vision Feed", processing_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except Exception as e:
                self._set_status(f"Runtime Exception: {e}")

    def request_action(self, cmd_packet: dict, timeout: float = 1.5) -> tuple:
        try:
            self._result_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._cmd_queue.put_nowait(cmd_packet)
        except queue.Full:
            return ("TIMEOUT", "NONE")
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return ("TIMEOUT", "NONE")


# --- 2. SINGLETON INITIALIZATION ---
@st.cache_resource
def get_vision_system():
    system = IndustrialVisionSystem()
    system.start_hardware()
    return system

vision_sys = get_vision_system()

# --- 3. STREAMLIT UI LAYOUT ---
st.title("📐 Real-World Calibrated Sorting System")

status = vision_sys.get_status()
st.sidebar.header("⚙️ Core Controls")
st.sidebar.metric("Current Scale Ratio", f"{round(vision_sys.get_pixels_per_mm(), 2)} px/mm")

# --- CALIBRATION WORKBENCH IN SIDEBAR ---
with st.sidebar.expander("📏 1-Step Dimension Calibration", expanded=True):
    st.write("Place an item of a known size under the camera lens to map pixels to real millimeters.")
    known_size = st.number_input("Real Object Length (mm)", min_value=1.0, max_value=500.0, value=25.4, step=0.1)
    if st.button("📐 Recalibrate System Scale", use_container_width=True):
        res, act = vision_sys.request_action({"type": "CALIBRATE", "target_mm": known_size})
        if res == "SUCCESS":
            st.success("Calibration complete!")
        else:
            st.error("Calibration failed: Stage empty.")

col_control, col_database = st.columns([1, 2])

with col_control:
    st.subheader("🎓 Register Component")
    with st.form("train_form"):
        new_name = st.text_input("Product Label", placeholder="e.g., Small_Bolt_M5")
        target_slot = st.number_input("Target Sorting Bin", min_value=1, max_value=12, value=1)
        
        # SENSITIVITY FILTER: Lets you deal with highly similar shapes safely
        tolerance_percentage = st.slider(
            "Matching Window Tolerance", 
            min_value=0.02, max_value=0.20, value=0.10, step=0.01,
            help="Lower values (e.g. 0.04 / 4%) prevent similar shapes from getting misidentified!"
        )
        
        submit = st.form_submit_button("📥 Lock Component Profile", use_container_width=True)

        if submit and new_name:
            res, act = vision_sys.request_action({
                "type": "TRAIN", "name": new_name, "slot": int(target_slot), "tolerance": tolerance_percentage
            })
            if res == "SUCCESS":
                st.toast(f"Profile '{new_name}' locked in millimeters!", icon="✅")
            elif res == "ERROR_EMPTY":
                st.error("No object detected to capture profile metrics.")

with col_database:
    st.subheader("📊 Calibrated Profile Directory")
    with vision_sys.product_db_lock:
        db_copy = dict(vision_sys.product_db)

    if db_copy:
        df = pd.DataFrame.from_dict(db_copy, orient="index")
        st.dataframe(df[["target_slot", "expected_width_mm", "expected_height_mm", "aspect_ratio", "tolerance"]], use_container_width=True)
    else:
        st.info("No profiles registered. Calibrate your scale, place an item, and register it above.")

    if st.button("🔄 Sync View", use_container_width=True):
        st.rerun()