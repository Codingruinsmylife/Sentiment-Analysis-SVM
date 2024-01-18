# Sentiment Analysis with SVM
## Overview
Sentiment Analysis using SVM is a robust project developed to harness the capabilities of machine learning for sentiment classification in magazine subscription reviews. By employing the Support Vector Machine (SVM) algorithm, the system categorizes reviews into positive, negative, or neutral sentiments. This project addresses the increasing demand for automated sentiment analysis, providing businesses with valuable insights into customer satisfaction, enabling them to make informed decisions based on customer feedback.

## Project Objectives
The main objectives of this project are: <br>
1. **Data Preprocessing** <br>
The initial step involves processing the raw text data extracted from magazine subscription reviews. This includes handling missing or irrelevant information, normalizing text by converting it to lowercase, and implementing techniques such as removing stopwords and stemming to ensure the model's efficiency.
2. **Model Training** <br>
The project leverages the Support Vector Machine algorithm, a powerful tool for binary classification tasks, to learn intricate patterns and relationships within the preprocessed text data. The linear kernel is chosen for its simplicity and proven success in text classification tasks.
3. **Performance Evaluation** <br>
To gauge the model's effectiveness, it undergoes rigorous evaluation on a separate test set. Metrics such as accuracy, precision, recall, and F1-score are computed to provide a comprehensive understanding of the model's performance. This step is crucial for ensuring the reliability of sentiment predictions in real-world scenarios.

## Key Components
This project is built upon several key components that collectively contribute to its functionality:
1. **Data Loading and Preprocessing** <br>
The raw dataset is loaded and converted into a structured Pandas DataFrame. The text data undergoes preprocessing steps such as converting to lowercase, removing stopwords, and stemming to prepare it for analysis.
2. **Sentiment Mapping** <br>
Ratings (1-5) are mapped to sentiment labels ("Positive," "Neutral," "Negative"). This step is crucial for converting numerical ratings into sentiment categories, enabling supervised learning.
3. **TF-IDF Vectorization** <br>
Term Frequency-Inverse Document Frequency (TF-IDF) is employed to vectorize the preprocessed text data. This technique transforms raw text into a numerical format suitable for machine learning models.
4. **SVM Classifier Training** <br>
The linear kernel Support Vector Machine is trained on the vectorized text data to learn patterns and relationships. The classifier is optimized for sentiment analysis tasks.
5. **Model Evaluation** <br>
The trained SVM classifier is rigorously evaluated on a separate test set to measure its performance. Metrics such as accuracy and classification reports are generated to assess the model's effectiveness.
6. **User Interaction** <br>
The project provides a function **'predict_sentiment'** that allows users to input custom text and receive sentiment predictions. This feature enhances the project's practical utility and user-friendliness.

## Why Use SVM Algorithm?
1. **High-Dimensional Data Handling** <br>
SVMs are particularly effective in handling high-dimensional data. In sentiment analysis, the feature space can be vast, with each unique word representing a dimension. SVMs excel at finding the optimal hyperplane to separate classes in such spaces.
2. **Binary Classification Expertise** <br>
SVMs are designed for binary classification tasks, making them well-suited for sentiment analysis, which involves distinguishing between positive, negative, and neutral sentiments. The linear kernel facilitates a clear decision boundary between classes.
3. **Robust Generalization** <br>
SVMs aim to find the optimal margin, ensuring robust generalization to unseen data. This is crucial in sentiment analysis, where the model needs to make accurate predictions on new reviews not seen during training.

## Dataset
The dataset for this project is sourced in JSON GZ format, containing magazine subscription reviews. The data undergoes loading, conversion to a Pandas DataFrame, and preprocessing to extract relevant information for sentiment analysis.

## Getting Started
To embark on this sentiment analysis journey, follow these steps:
1. **Install Dependencies** <br>
Make sure you have the required dependencies installed. Run the following command in your Python environment:
```bash
pip install pandas numpy nltk scikit-learn
```
2. **Download Dataset** <br>
Download the dataset file "Magazine_Subscription.json.gz."
3. **Execute code** <br>
Run the provided code in a Python environment. Ensure that the dataset file is in the same directory as your code file.

## Usage
To unleash the power of the model:
1. Execute the code to load and preprocess the dataset.
2. Train the SVM classifier on the training set.
3. Evaluate the model on the test set.
4. Leverage the **'predict_sentiment'** function to predict sentiment on custom texts.

## Contributing
Contributions to this project are enthusiastically welcomed. Whether it's enhancing functionality or fixing bugs, feel free to submit issues or pull requests to propel the project forward.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Codingruinsmylife/Sentiment-Analysis-SVM/blob/main/LICENSE.txt) file for details.

