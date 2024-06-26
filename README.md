# 🖼️ Image Caption Generator

## Overview
The Image Caption Generator is a machine learning project that generates captions for images using deep learning techniques. This project utilizes a combination of Convolutional Neural Networks (CNNs) for image feature extraction and Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) networks, for sequence generation.

## ✨ Features
- 📸 Extracts features from images using a pre-trained CNN (e.g., InceptionV3, VGG16).
- 📝 Generates captions using an LSTM-based RNN.
- 📂 Supports training on custom datasets.
- 🌐 Provides a user interface for uploading images and displaying generated captions.

## 🚀 Getting Started

### Prerequisites
- 🐍 Python 3.7 or higher
- 📦 TensorFlow 2.x
- 🧮 NumPy
- 🐼 Pandas
- 📊 Matplotlib
- 🖼️ Pillow
- 🖥️ Flask (for the web interface)

### 📥 Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/image-caption-generator.git
    cd image-caption-generator
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### 📚 Dataset
For training, we use the [MS COCO dataset](https://cocodataset.org/#home). Download and extract the dataset.

### 🏋️‍♂️ Training the Model
1. Extract image features using a pre-trained CNN:
    ```python
    python extract_features.py --dataset_path path/to/coco/images --output_path path/to/save/features
    ```

2. Preprocess the captions and prepare the dataset:
    ```python
    python preprocess_captions.py --annotations_path path/to/coco/annotations --output_path path/to/save/preprocessed/data
    ```

3. Train the model:
    ```python
    python train.py --features_path path/to/saved/features --captions_path path/to/preprocessed/data --output_model_path path/to/save/model
    ```

### 🖼️ Generating Captions
To generate captions for new images:
1. Run the caption generation script:
    ```python
    python generate_caption.py --image_path path/to/image --model_path path/to/saved/model
    ```

2. Or, use the web interface:
    ```sh
    python app.py
    ```
    Open your browser and go to `http://127.0.0.1:5000` to upload an image and get the caption.

## 📂 Project Structure
- `extract_features.py`: Script to extract image features using a pre-trained CNN.
- `preprocess_captions.py`: Script to preprocess captions and prepare the dataset.
- `train.py`: Script to train the image captioning model.
- `generate_caption.py`: Script to generate captions for new images.
- `app.py`: Flask application for the web interface.
- `model.py`: Contains the model architecture and related functions.
- `utils.py`: Utility functions for data processing and model operations.

## 📑 References
- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
- [MS COCO Dataset](https://cocodataset.org/#home)

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements
- The TensorFlow and Keras teams for their excellent deep learning libraries.
- The creators of the MS COCO dataset for providing a valuable resource for training image captioning models.
