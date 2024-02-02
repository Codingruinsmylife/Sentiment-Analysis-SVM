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
1. **Execute the code to load and preprocess the dataset.**
```bash
# Load the JSON GZ dataset
def load_json_gz_dataset(filename):
    # ... (code for loading)

# Load and preprocess the JSON GZ dataset
dataset = load_json_gz_dataset("Magazine_Subscriptions.json.gz")

# Convert the dataset to a Pandas DataFrame
df = pd.DataFrame(dataset)

# Display some sample data
print("Sample Data:")
print(df.head())
```
2. **Train the SVM classifier on the training set.**
```bash
# Split the data into training and testing sets
X = df["reviewText"]
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train an SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train_tfidf, y_train)
```
3. **Evaluate the model on the test set.**
```bash
# Evaluate the model
y_pred = svm_classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Model Evaluation:")
print("Accuracy:", accuracy)
print("Classification Report:", report)
```
4. **Leverage the **'predict_sentiment'** function to predict sentiment on custom texts.**
```bash
# Test model's correctness
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
    sentiment = svm_classifier.predict(text_tfidf)
    return sentiment[0]

sample_texts = [
    "This product is amazing and I love it!",
    "The service was terrible and I'm highly disappointed.",
    "Neutral sentiment for this one."
]

print("\nTesting the Model on Sample Texts:")
for text in sample_texts:
    sentiment = predict_sentiment(text)
    print(f"Text: '{text}'")
    print(f"Predicted Sentiment: {sentiment}\n")
```
## Contributing
We appreciate your interest in contributing to the Time Series Analysis Model project. Whether you are offering feedback, reporting issues, or proposing new features, your contributions are invaluable. Here's how you can get involved:
### How to Contribute
1. **Issue Reporting**
   * If you encounter any issues or unexpected behavior, please open an issue on the project.
   * Provide detailed information about the problem, including steps to reproduce it.
2. **Feature Requests**
   * Share your ideas for enhancements or new features by opening a feature request on GitHub.
   * Clearly articulate the rationale and potential benefits of the proposed feature.
3. **Pull Requests**
   * If you have a fix or an enhancement to contribute, submit a pull request.
   * Ensure your changes align with the project's coding standards and conventions.
   * Include a detailed description of your changes.
  
## License
The Time Series Analysis Model project is open-source and licensed under the [MIT License](LISENCE). By contributing to this project, you agree that your contributions will be licensed under this license. Thank you for considering contributing to our project. Your involvement helps make this project better for everyone. <br><br>
**Haev Fun!** 🚀

