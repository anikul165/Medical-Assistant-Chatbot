# Medical Question-Answering Chatbot

## Overview

This project is a Medical Question-Answering Chatbot designed to respond to user queries related to medical diseases. The chatbot is trained on a dataset containing pairs of questions and answers about various medical conditions. It uses an LSTM model for understanding and generating responses to user queries.

## Features

- **Natural Language Processing**: Preprocessing steps include tokenization, lemmatization, and sequence padding.
- **Deep Learning**: An LSTM-based model is used to train on the preprocessed dataset.
- **Interaction**: The chatbot can interact with users and provide answers to their medical queries.
- **Wikipedia Integration**: The chatbot can compare its answers with Wikipedia responses for validation.

## Requirements

Ensure you have the following Python packages installed:

- numpy
- pandas
- torch
- nltk
- sklearn
- requests

You can install the necessary packages using:

```bash
pip install numpy pandas torch nltk scikit-learn requests
```

## Dataset
The dataset used for training the chatbot is provided in a CSV file with two columns: questions and answers. The dataset should be preprocessed to remove duplicates and handle missing values.

## Preprocessing
The preprocessing steps include:

* Converting text to lowercase.
* Removing punctuation.
* Tokenizing the text.
* Lemmatizing the tokens.
* Encoding the text into numerical values.
* Padding sequences to ensure uniform input size.

## Model Architecture
The model is built using an LSTM (Long Short-Term Memory) network. The architecture includes:

* An embedding layer to convert words into vectors.
* LSTM layers to capture the temporal dependencies in the data.
* A fully connected layer for the final classification.

## Training
The model is trained using Cross-Entropy Loss and the Adam optimizer. The training process includes:

* Splitting the dataset into training, validation, and test sets.
* Using DataLoader for batching and shuffling the data.
* Training the model for a specified number of epochs.
* Evaluating the model's performance using accuracy, precision, recall, and F1-score.

## Usage
### Training the Model
To train the model, run the script:
```bash
python train.py
```

## Chatbot Interaction
To interact with the chatbot, use the following function:
```bash
response = chatbot_response("What is diabetes?")
print(response)
```

## Comparing with Wikipedia
To compare the chatbot's response with Wikipedia, use the compare_with_wikipedia function:
```bash
wiki_similarity = compare_with_wikipedia("What is diabetes?", chatbot_response("What is diabetes?"))
print(f'Similarity with Wikipedia: {wiki_similarity}')
```

### Example Interactions
```bash
print(chatbot_response("What is diabetes?"))
print(chatbot_response("Tell me about asthma"))
print(chatbot_response("What are the symptoms of flu?"))
```

## Performance Metrics
The model's performance is evaluated using:

* Accuracy: The percentage of correctly predicted answers.
* Precision: The ratio of true positive predictions to the total predicted positives.
* Recall: The ratio of true positive predictions to the total actual positives.
* F1-Score: The harmonic mean of precision and recall.

## Assumptions
* The dataset is representative of the questions the chatbot will encounter.
* The preprocessing steps are appropriate for the data.
* The LSTM model's architecture is suitable for the task.

## Potential Improvements
* Enhanced Preprocessing: Incorporate advanced NLP techniques like BERT embeddings.
* Model Tuning: Experiment with different model architectures and hyperparameters.
* Data Augmentation: Expand the dataset with more diverse questions and answers.
* Improved Similarity Measures: Use advanced methods for comparing chatbot responses with external sources like Wikipedia.

## Contributors
Aniket Kulkarni
