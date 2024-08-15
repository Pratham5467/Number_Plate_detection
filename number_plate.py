import cv2
import os
import time
import streamlit as st
import numpy as np

def main():
    st.title("Number Plate Detection")

    harcascade = "model/haarcascade_russian_plate_number.xml"

    # Create directories to save the frames and plates
    for directory in ["frames", "plates"]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    st.sidebar.header("Settings")
    max_frames = st.sidebar.slider("Max Frames", 1, 500, 200)
    max_runtime = st.sidebar.slider("Max Runtime (seconds)", 1, 120, 30)
    min_area = st.sidebar.slider("Min Area", 100, 1000, 500)

    start_button = st.sidebar.button("Start Detection")

    if start_button:
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)  # width
        cap.set(4, 480)  # height

        count = 0
        start_time = time.time()

        # Placeholder for video feed
        stframe = st.empty()
        
        # Placeholder for status
        status_text = st.empty()

        while count < max_frames and (time.time() - start_time) < max_runtime:
            success, img = cap.read()

            if not success:
                st.error("Failed to capture frame")
                break

            plate_cascade = cv2.CascadeClassifier(harcascade)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

            img_roi = None
            for (x,y,w,h) in plates:
                area = w * h

                if area > min_area:
                    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

                    img_roi = img[y: y+h, x:x+w]
                    cv2.imwrite(f"frames/roi_{count}.jpg", img_roi)

            # Save the frame
            cv2.imwrite(f"frames/frame_{count}.jpg", img)
            
            # Save plate if detected
            if img_roi is not None:
                cv2.imwrite(f"plates/scaned_img_{count}.jpg", img_roi)
                status_text.text(f"Plate saved as plates/scaned_img_{count}.jpg")

            count += 1

            # Display the frame
            stframe.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

            # Small delay to reduce CPU usage
            time.sleep(0.1)

        cap.release()
        cv2.destroyAllWindows()
        st.success(f"Program ended after processing {count} frames")
        st.info(f"Total runtime: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()