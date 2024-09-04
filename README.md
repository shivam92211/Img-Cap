# Image Caption Generator

## Overview

This project is an Image Caption Generator that uses deep learning techniques to automatically generate descriptive captions for images. It consists of two main components:

1. A machine learning model for training and prediction
2. A Streamlit web application for user interaction

The system uses a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to generate captions. It leverages the VGG16 model for image feature extraction and an LSTM network for sequence generation.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
   - [Training the Model](#training-the-model)
   - [Making Predictions](#making-predictions)
   - [Running the Streamlit App](#running-the-streamlit-app)
3. [Code Structure](#code-structure)
4. [Model Architecture](#model-architecture)
5. [Dataset](#dataset)
6. [Evaluation](#evaluation)
7. [Streamlit App Functionality](#streamlit-app-functionality)
8. [Contributing](#contributing)
9. [License](#license)

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/shivam92211/image-caption-generator.git
   cd image-caption-generator
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model, run the following file:

Image-Caption.ipynb

This script will:
- Load and preprocess the Flickr8k dataset
- Extract features from images using VGG16
- Prepare the text data
- Train the caption generation model
- Save the trained model and necessary artifacts

### Running the Streamlit App

To launch the Streamlit web application, run:

```
streamlit run app.py
```

This will start the web server, and you can access the application by opening the provided URL in your web browser.

## Code Structure

The project is organized as follows:

```
image-caption-generator/
│
├── Image-Caption.ipynb          # Script for training the model
├── main.py                  # Streamlit application
├── model.keras             # Saved model file
├── features.pkl            # Extracted image features
├── mapping.pkl             # Image-to-captions mapping
├── requirements.txt        # Project dependencies
├── README.md               # This file
└── flickr8k/               # Dataset directory
    ├── Images/             # Image files
    └── captions.txt        # Image captions
```

## Model Architecture

The model architecture consists of two main parts:

1. **Encoder**: Uses VGG16 (pre-trained on ImageNet) to extract image features.
2. **Decoder**: An LSTM network that generates captions based on the image features and previously generated words.

The model is trained to minimize categorical cross-entropy loss.

## Dataset

This project uses the Flickr8k dataset, which contains 8,000 images, each with five different captions. The dataset is split into training (6,000 images) and testing (2,000 images) sets.

## Evaluation

The model's performance is evaluated using the BLEU (Bilingual Evaluation Understudy) score, which measures the similarity between the generated captions and the ground truth captions.

## Streamlit App Functionality

The Streamlit app provides a user-friendly interface for interacting with the trained model:

1. Users can upload an image file (JPG format).
2. The app preprocesses the image and extracts features using VGG16.
3. The caption generation model then produces a description for the image.
4. The uploaded image and the generated caption are displayed to the user.

## Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature-name`)
6. Create a new Pull Request

## License

[Specify your license here, e.g., MIT License, Apache License 2.0, etc.]