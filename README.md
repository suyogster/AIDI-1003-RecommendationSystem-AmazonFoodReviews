# Amazon Food Reviews Recommendation System

This project aims to build a hybrid recommendation system for Amazon food reviews using collaborative filtering and content-based filtering techniques. The system is implemented using Python and various libraries for data manipulation, visualization, text processing, and machine learning.

## Project Structure

- `requirements.txt`: Lists the dependencies required for the project.
- `app.py`: Contains the Flask web application to serve the recommendation system.
- `recommendationsystem-amazonfoodreviews.ipynb`: Jupyter notebook with the complete process of data loading, preprocessing, model building, and evaluation.

## Dependencies

The project requires the following libraries:

- numpy
- pandas
- scikit-learn
- nltk
- matplotlib
- seaborn
- scikit-surprise
- Flask

## Process Overview

### 1. Library Imports and Configuration

The necessary libraries are imported, and some parameters are configured for the project. This includes libraries for data manipulation (`pandas`, `numpy`), visualization (`matplotlib`, `seaborn`), text processing (`nltk`), and machine learning (`sklearn`, `surprise`).

### 2. Data Loading

The dataset is loaded from a CSV file. A function is defined to preprocess text by tokenizing and removing stopwords. The dataset is optionally sampled to speed up processing.

### 3. Data Cleaning & Preprocessing

Duplicates are removed, and users and items with fewer than a specified number of reviews are filtered out. The `Time` column is converted to datetime format, and the `HelpfulnessRatio` is calculated. The cleaned data is stored in `ratings_df`.

### 4. Exploratory Data Analysis (EDA)

Basic information and statistics about the dataset are printed. Various visualizations are created to understand the distribution of ratings, number of reviews per user and product, reviews over time, and the relationship between helpfulness and score.

### 5. Feature Engineering for Hybrid Approach

Text data is combined and preprocessed to create a TF-IDF matrix. Item similarity is calculated using cosine similarity. Mappings between product IDs and indices are created.

### 6. Model Selection & Building (Collaborative Filtering)

The collaborative filtering model is set up using the `SVDpp` algorithm from the `surprise` library. The data is split into training and test sets, and the model is trained and cross-validated.

### 7. Hyperparameter Tuning

Hyperparameter tuning is performed for the `SVDpp` model using grid search to find the best parameters based on RMSE and MAE.

### 8. Model Evaluation

The trained model is evaluated on the test set, calculating RMSE and MAE. A function is defined to calculate precision and recall at k.

### 9. Hybrid Recommendation Logic

A function is defined to generate hybrid recommendations for a user by combining collaborative filtering predictions with content-based similarity scores.

### 10. Testing the Recommendation Function

The hybrid recommendation function is tested for a sample user, and the top 5 recommended items are printed.

### 11. Export Model Artifacts

The trained model and related artifacts are exported to a file for later use.

## Flask Web Application

The `app.py` file contains the Flask web application to serve the recommendation system. It includes the following routes:

- `/`: Renders the home page.
- `/recommend`: Accepts a POST request with a user ID and returns the top 5 recommendations for the user.
- `/random_recommend`: Returns the top 5 recommendations for a randomly selected user.

## Running the Application

To run the Flask web application, execute the following command:

```bash
python app.py