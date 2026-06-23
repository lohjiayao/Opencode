import cv2
import numpy as np
import imutils

# 1. This function stays outside
def get_camera_index_by_name(target_name: str) -> int:
    try:
        import pygrabber.dshow_graph as dshow
        devices = dshow.FilterGraph().get_input_devices()
        for i, name in enumerate(devices):
            if target_name.lower() in name.lower():
                return i
    except Exception:
        pass
    return 0

def main():
    PREFERRED_CAMERA = "c922" 
    index = get_camera_index_by_name(PREFERRED_CAMERA)
    
    # Define 'cap' INSIDE main
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read() # Now 'cap' is defined in this scope!
        if not ret:
            break

        # --- AI FEATURE DETECTION ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            if cv2.contourArea(c) < 1000:
                continue

            # This calculates Orientation (angle) and Size (w, h)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # Draw the features
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
            
            # Show Angle (Orientation)
            angle = rect[2]
            cv2.putText(frame, f"Angle: {int(angle)}deg", (int(box[0][0]), int(box[0][1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('AI Vision Test', frame)
        cv2.imshow('What the AI Sees (Edges)', edged)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# This starts the whole thing
if __name__ == "__main__":
    main()