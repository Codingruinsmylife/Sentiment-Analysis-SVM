# Sentiment Analysis with Support Vector Machine
## Overview
This project demonstrates sentiment analysis using Support Vector Machines (SVM) with a focus on text data. Sentiment analysis, also known as opinion mining, involves determining the sentiment or emotional tone of text, which can be valuable for various applications, such as social media monitoring, product reviews, and more.

In this project, we perform sentiment analysis on a dataset of product reviews using a linear SVM classifier. We also conduct hyperparameter tuning to optimize the model's performance.

## Table of Contents
* ### Prerequisites
* ### Project Structure
* ### Usage
* ### Data Preprocessing
* ### Model Training
* ### Hyperparameter Tuning
* ### Evaluation
* ### Results
* ### Contributing
* ### License

## Prerequisites
Before running the code, you need to have the following installed:
* Python (version 3.6 or later)
* Required Python libraries (install using 'pip'):
  * 'pandas'
  * 'numpy'
  * 'nltk'
  * 'scikit-learn'

## Project Structure
The project structure is as follows:
* 'data/': Contains the dataset files.
* 'code/': Contains the python code.
  * 'sentiment_analysis.ipynb': The main code for sentiment analysis.
* 'results/': Stores the output and model files.
* 'README.md': The project documentation (this file).

## Usage
To run the project, follow these steps:
1. Clone the repository:
   ```
   git clone https://github.com/codingruinsmylife/sentiment-analysis-svm.git
   cd sentiment-analysis-svm
   ```
2. Install the required libraries:
   ```
   !pip3 install -r requirement.txt
   ```
3. Run the code:
   ```
   python code/sentiment_analysis.ipynb
   ```
## Data Preprocessing
The dataset is preprocessed as follows:
* Text data is converted to lowercase.
* Stopwords are removed.
* Stemming is applied to reduce words to their root form.

## Model Training
* The dataset is split into training and testing sets.
* Text data is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency).
* A linear SVM classifier is trained on the vectorized text data.

## Hyperparameter Tuning
* I perform hyperparameter tuning to optimize the SVM model.
* I vary the regularization parameter ('C') while keeping the kernel type fixed as 'linear' as the linear kernel is often a good choice for text classification tasks.
* 'GridSearchCV' is used to find the best combination of hyperparameters.

## Evaluation
* The model is evaluated using the testing set.
* Accuracy and classification report are provided to assess model performance.

## Results
The project results include the best hyperparameters ('C') and the best SVM classifier based on hyperparameter tuning. These results are obtained using 'GridSearchCV'

## Contributing
If you would like to contribute to this project, please follow the standard procedures for open-source contributions. Feel free to create a pull request or open an issue.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgement
* This project is inspired by the need for sentiment analysis in natural language processing tasks.
* The code structure and README template are based on best practices for organizing and documenting Python projects.
Thank you for using this sentiment analysis project! I hope it's useful for your text data analysis tasks.
