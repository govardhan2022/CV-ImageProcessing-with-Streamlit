import streamlit as st
import cv2
import numpy as np

def perform_edge_detection(image, method):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == 'Edge Detection':
        edges = cv2.Canny(gray, 50, 150)
    elif method == 'Corner Detection':
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        edges = np.zeros_like(gray)
        edges[corners > 0.01 * corners.max()] = 255
    elif method == 'Canny':
        edges = cv2.Canny(gray, 50, 150)
    elif method == 'DoG':
        blurred1 = cv2.GaussianBlur(gray, (5, 5), 0)
        blurred2 = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = blurred1 - blurred2

    return edges

def main():
    st.title("Image Processing with Streamlit")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Convert the BytesIO object to a numpy array
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            if image is not None:
                st.image(image, caption="Uploaded Image", use_column_width=True)

                method = st.selectbox("Select Method", ['Edge Detection', 'Corner Detection', 'Canny', 'DoG'])

                if st.button("Process"):
                    edges = perform_edge_detection(image, method)
                    st.image(edges, caption=f"{method} Result", use_column_width=True, channels='GRAY')
            else:
                st.error("Error loading the image. Please make sure it is a valid image file.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()