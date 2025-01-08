# Youtube-Spam-Detection

# STA314
This repository contains the final project of STA314, the Statistical Methods in Machine Learning - I, from University of Toronto. 

# YouTube-Spam-Detection

## Overview

YouTube, with its vast reach and billions of daily interactions, serves as a significant platform for information and entertainment. However, this popularity has attracted spammers who post irrelevant or promotional comments that disrupt the user experience. The objective of this project is to develop an automated classifier that accurately flags spam comments, improving the quality of interactions on YouTube videos. This project leverages machine learning techniques, including Support Vector Machines (SVM) and neural networks, to identify spam comments with a high degree of accuracy.

## Project Highlights

- **Objective**: To classify YouTube comments as "spam" or "non-spam" to improve content quality on the platform.
- **Dataset**: Kaggle dataset containing 1369 labeled YouTube comments, present in file `test.csv`.
- **Approach**: Text-based feature engineering using TF-IDF, experimenting with both SVM and a basic neural network model, with final selection of SVM after hyperparameter tuning.
- **Report**: A comprehensive project report documenting the methodology, results, and insights is stored in the file `Report`.
- **Code**: All models and code are located in the file `scratchpad`.

## Methodology

### 1. Data Preprocessing

- Verified dataset integrity by checking for missing values—none were found, ensuring a clean start.
- Utilized TF-IDF vectorization to convert comment text into numerical feature vectors, capturing term relevance and frequency while reducing dimensionality. This approach produced a vocabulary of 2821 unique words.

### 2. Feature Engineering with TF-IDF

- Employed the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to transform each comment into a meaningful vector representation, aiding in distinguishing spam comments.

### 3. Model Development

- **Models**: We initially tested a basic neural network model and Support Vector Machine (SVM), ultimately selecting the SVM due to its superior performance after hyperparameter tuning.
- **Training and Testing**: Split the data into 80% training and 20% testing sets to ensure robust model evaluation.
- **Hyperparameter Tuning**: The SVM model was fine-tuned to optimize performance, further improving its ability to differentiate between spam and non-spam comments.

### 4. Performance Evaluation

- The model's performance was evaluated using accuracy, precision, recall, and F1-score metrics, with the final SVM model delivering strong results.

## Results

- The final SVM model, after hyperparameter tuning, achieved high accuracy and demonstrated strong predictive performance in distinguishing between legitimate comments and spam.
- The models were evaluated for real-time prediction capability, allowing users to input custom comments and receive an instant spam or non-spam classification.

## Key Learnings

- **Model Selection and Tuning**: The project highlighted the importance of experimenting with multiple models and performing hyperparameter tuning to optimize performance. SVM outperformed the neural network model, making it the best choice for this task.
- **Feature Engineering for Text Data**: Understanding how TF-IDF representations enhanced the model's ability to classify spam comments was crucial in achieving high accuracy.
- **Report and Code Documentation**: Keeping detailed records of the methodology and results in the project report (`Report`) and maintaining all models and code in `scratchpad` facilitated easier project management and reproducibility.

## Conclusion

This project demonstrates the application of machine learning in social media moderation, specifically YouTube comment spam detection. The final SVM classifier provides a robust solution for identifying spam, which can be adapted for real-world use. Future work could explore more advanced techniques such as deep learning models or ensemble methods to further improve the system's performance.

## Future Directions

- **Advanced Techniques**: I plan to experiment with more advanced text representation techniques, such as word embeddings or transformer models, to capture deeper semantic nuances in comments.
- **Multi-Label Classification**: I will also explore extending the project to handle multi-label classification for more comprehensive comment moderation (e.g., detecting offensive content, irrelevant content, etc.).
