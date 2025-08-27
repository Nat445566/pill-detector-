\# 💊 Smart Pill Counting System with Streamlit and OpenCV



!\[Pill Counter App Screenshot](https://user-images.githubusercontent.com/\[Nat445566]/\[pill-detector-]/screenshot.png)



A user-friendly web application that allows users to count pills from an uploaded image. By interactively selecting a sample pill, the system leverages OpenCV to automatically detect and count all similar pills, providing a quick and accurate tally.



This project was developed as a practical application of the concepts learned in the "BMDS2133 Image Processing" course.



---



\## ✨ Features



\-   \*\*Interactive Image Upload:\*\* Supports JPG, JPEG, and PNG formats.

\-   \*\*Drawable Canvas for ROI Selection:\*\* Users can easily draw a rectangle on the uploaded image to select a sample pill.

\-   \*\*Automated Object Counting:\*\* Employs a robust image processing pipeline to accurately count objects similar in size to the selected sample.

\-   \*\*Clear Visual Feedback:\*\* The final image displays the counted pills highlighted with green boxes and a clear total count.



---



\## 💻 How to Run Locally



Follow these instructions to get the project running on your local machine.



\### Prerequisites



\-   Python 3.8+

\-   `pip` (Python package installer)



\### Installation



1\.  \*\*Clone the repository:\*\*

&nbsp;   ```bash

&nbsp;   git clone https://github.com/\[Your GitHub Username]/\[Your Repo Name].git

&nbsp;   cd \[Your Repo Name]

&nbsp;   ```



2\.  \*\*Install the required libraries using `requirements.txt`:\*\*

&nbsp;   ```bash

&nbsp;   pip install -r requirements.txt

&nbsp;   ```



\### Running the Application



1\.  \*\*Navigate to the project directory in your terminal.\*\*



2\.  \*\*Run the Streamlit application:\*\*

&nbsp;   ```bash

&nbsp;   streamlit run app.py

&nbsp;   ```



The application should automatically open in a new tab in your web browser.



---



\## ⚙️ Technology Stack



\-   \*\*Backend:\*\* Python

\-   \*\*Web Framework:\*\* Streamlit

\-   \*\*Image Processing:\*\* OpenCV

\-   \*\*Numerical Operations:\*\* NumPy

\-   \*\*Interactive Canvas:\*\* `streamlit-drawable-canvas`



---



\## 🧠 How It Works: The Image Processing Pipeline



The core logic of the pill counter follows these steps:



1\.  \*\*Image Preprocessing:\*\* The uploaded image is first converted to grayscale to simplify analysis. A Gaussian blur is then applied to reduce noise and smooth edges.

2\.  \*\*Segmentation:\*\* Using \*\*Otsu's Binarization\*\*, the system automatically creates a binary (black and white) mask to separate the pills (foreground) from the background.

3\.  \*\*Noise Removal:\*\* \*\*Morphological Operations\*\* (specifically Opening and Closing) are used to clean the binary mask by removing small noise specks and filling in tiny holes within the pills.

4\.  \*\*Contour Detection:\*\* The application finds the contours (outlines) of all objects in the cleaned mask.

5\.  \*\*Filtering \& Counting:\*\* Each contour is filtered based on its area. Only contours with an area similar to the user-selected sample pill are considered valid. The final count is the number of valid contours found.



---



\## 📜 License



This project is licensed under the MIT License. See the `LICENSE` file for details.


