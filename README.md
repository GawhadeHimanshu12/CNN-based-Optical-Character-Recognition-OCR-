# CNN-based Optical Character Recognition (OCR)

This project implements an Optical Character Recognition (OCR) system using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The script trains a model to recognize individual alphanumeric characters (0-9, A-Z, a-z) from a dataset and then uses the trained model to predict text from a given image file.

## Features

* **CNN Model:** Utilizes a Convolutional Neural Network for feature extraction and classification.
* **Character Set:** Recognizes 62 alphanumeric classes (0-9, uppercase A-Z, lowercase a-z).
* **Data Preprocessing:** Includes image loading, resizing (28x28), grayscale conversion, and normalization.
* **Data Augmentation:** Uses `ImageDataGenerator` to apply random transformations (rotation, shifts, shear, zoom) during training to improve model robustness.
* **Training:**
    * Splits data into training and validation sets.
    * Uses Adam optimizer and categorical cross-entropy loss.
    * Implements Early Stopping based on validation accuracy to prevent overfitting and save the best model weights.
    * Visualizes training and validation accuracy/loss curves using Matplotlib.
* **Prediction:**
    * Includes a function (`predict_text_from_image`) to process a new image.
    * Performs basic character segmentation using OpenCV contour detection.
    * Preprocesses segmented characters to match the model's input requirements.
    * Predicts characters and reconstructs the text.
* **Evaluation:** Calculates and displays detailed metrics (Accuracy, Precision, Recall, F1-score) and a confusion matrix on the test set after training.

## Dataset

* The script expects the training data to be organized in a specific directory structure under `font/Fnt/`.
* Each character class should have its own subfolder named `SampleXXX`, where `XXX` is a three-digit number from `001` to `062`.
    * `Sample001` to `Sample010`: Digits 0 to 9
    * `Sample011` to `Sample036`: Uppercase letters A to Z
    * `Sample037` to `Sample062`: Lowercase letters a to z
* Images within these folders should be in `.png` format.
* The Chars74k dataset's "Fnt" (Font) subset is suitable for this structure. You will need to download and place it accordingly.

## Performance (Example)

This section describes the typical performance achievable with this model on suitable datasets. The actual performance depends significantly on the specific training data used (e.g., `font/Fnt/`), the train/test split, and training duration.

Based on the CNN architecture, data augmentation, and training procedures implemented in this script, when run on a standard clean font dataset (like the Chars74k Fnt subset), a plausible result indicating how accurately the model predicts would be:

* **Overall Test Accuracy:** Approximately **93.5%**
* **Weighted Avg Precision:** Approximately **0.94**

*(Note: These are example figures representing reasonable results for this project setup under typical conditions. Your actual metrics, printed by the script after training, might differ based on your specific dataset and execution environment.)*

The script provides a detailed evaluation after training, allowing for a thorough assessment:
* Overall accuracy on the test set.
* A classification report showing **precision**, recall, and F1-score for each character class. This helps understand the accuracy for specific characters.
* A confusion matrix visualizing specific class confusions (e.g., common errors between '6' and 'b', or 'O' and '0').

This detailed output helps evaluate precisely how well the model performs and where potential weaknesses lie.

## Requirements

* Python 3.x
* TensorFlow (`tensorflow`)
* OpenCV (`opencv-python`)
* Scikit-learn (`scikit-learn`)
* NumPy (`numpy`)
* Matplotlib (`matplotlib`)
* Seaborn (`seaborn`) (for confusion matrix visualization)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    tensorflow
    opencv-python
    scikit-learn
    numpy
    matplotlib
    seaborn
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Prepare the Dataset:**
    * Download or create your character dataset.
    * Organize it in the `font/Fnt/` directory according to the structure described in the [Dataset](#dataset) section. If your dataset path is different, modify the `data_dir` variable in the script.
5.  **Prepare Prediction Image:**
    * Place the image you want to test prediction on (e.g., `hw.png`) in the project's root directory, or update the `image_path` variable in the script accordingly.

## Usage

1.  Ensure your dataset is correctly placed in `font/Fnt/` and the test image (`hw.png` or other) is available.
2.  Run the main script:
    ```bash
    python your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual name of your Python file).

Running the script will:
* Load and preprocess the data from the `data_dir`.
* Split the data into training and testing sets.
* Define and compile the CNN model.
* Train the model using data augmentation and early stopping. The console will show training progress per epoch.
* Display plots for training/validation accuracy and loss.
* Evaluate the model on the test set and print the accuracy, classification report (including precision), and display the confusion matrix.
* Load the specified `image_path`.
* Perform character segmentation and prediction on the image.
* Print the predicted text to the console.

## Model Architecture

The CNN consists of multiple blocks, each typically containing:
* `Conv2D` layers with ReLU activation for feature extraction.
* `BatchNormalization` for stabilizing training.
* `MaxPooling2D` for downsampling.
* `Dropout` for regularization.

These blocks are followed by:
* `Flatten` layer to convert 2D feature maps to 1D.
* `Dense` (fully connected) layers with ReLU activation.
* `BatchNormalization` and `Dropout`.
* Final `Dense` output layer with 62 units and `softmax` activation for multi-class classification.

Refer to the `model.summary()` output in the console when running the script for detailed layer information.

## Prediction Steps (`predict_text_from_image` function)

1.  Load the image in color and convert it to grayscale.
2.  Apply Gaussian Blur to reduce noise.
3.  Apply inverse binary thresholding to isolate potential characters (white on black background).
4.  Find external contours of the white regions.
5.  Sort contours from left to right based on their bounding box x-coordinate.
6.  For each valid contour (size check `w > 5 and h > 5`):
    * Extract the character ROI from the *original grayscale* image using the bounding box.
    * Resize the ROI to 28x28 pixels.
    * Normalize the pixel values (divide by 255.0).
7.  Stack the processed character images into a batch.
8.  Use the trained `model.predict()` to get class probabilities for each character.
9.  Determine the most likely class label using `np.argmax`.
10. Map the predicted numeric labels back to their corresponding characters (0-9, A-Z, a-z).
11. Concatenate the characters to form the final predicted text string.

## Limitations

* **Segmentation Sensitivity:** The contour-based character segmentation is basic and highly dependent on image quality. It may perform poorly on images with:
    * Noisy backgrounds.
    * Touching or overlapping characters.
    * Skewed or heavily stylized text.
    * Very small characters (might be filtered out).
* **Font Dependence:** The model's accuracy will be best on fonts similar to those in the training dataset.
* **Limited Character Set:** Only recognizes alphanumeric characters (0-9, A-Z, a-z). It does not support punctuation, special symbols, or other languages.

## Potential Future Improvements

* Implement more robust character segmentation techniques (e.g., projection profiles, connected components analysis with better filtering, or deep learning-based segmentation).
* Train on a more diverse dataset including various fonts and augmentations.
* Expand the character set to include punctuation and symbols.
* Experiment with different CNN architectures (e.g., ResNet, VGG variants).
* Perform hyperparameter tuning for potentially better performance.
* Implement techniques to specifically address confusion between visually similar characters.
#   C N N - b a s e d - O p t i c a l - C h a r a c t e r - R e c o g n i t i o n - O C R -  
 