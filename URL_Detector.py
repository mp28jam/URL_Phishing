# ===============================================================
# URL DETECTOR PROJECT
# ===============================================================

# Used to measure train time and run time
import time
# Used to process data
import pandas as pd
# Used to save and load models
import joblib
# Used for regular expression parsing
import re
# Used for numerical Computing
import numpy as np
# Used to convert URLS into numerical features
from sklearn.feature_extraction.text import TfidfVectorizer
# Used to split data into train and test sets
from sklearn.model_selection import train_test_split
# Used for Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
# Used to combine tfidf and numeric features
from scipy.sparse import hstack
# Used to compute class weights for class imbalance
from sklearn.utils.class_weight import compute_class_weight
# Used to parse URL components
from urllib.parse import urlparse
# Used to print precision/recall/F1
from sklearn.metrics import classification_report


# ===============================================================
# helper function extract_url_features()
# extract numeric URL features
# ===============================================================

def extract_url_features(df):

    # Input: df with a column 'URL'
    # Output: DataFrame with numeric features extracted from each URL

    df = df.copy()                                                              # Work on a df copy
    df["url_length"] = df["URL"].apply(len)                                     # Total characters in the URL
    df["num_digits"] = df["URL"].str.count(r"[0-9]")                            # Number of digits in the URL
    df["num_special"] = df["URL"].str.count(r"[-._@/%?=&]")                     # Number of common special characters
    df["num_subdomains"] = df["URL"].str.count(r"\.")                           # Number of '.' characters 
    # Mark as 1 if URL starts with an IP address else 0
    df["has_ip"] = df["URL"].apply(lambda x: 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}", x) else 0)
    # Length of domain, extracts the domain part of the URL and measures its length 
    df["domain_length"] = df["URL"].apply(lambda x: len(urlparse(x).netloc))
    # Ratio of digits to total length
    df["digits_ratio"] = df["num_digits"] / df["url_length"].replace(0, 1)

    # Return the numeric columns which are our new features
    return df[["url_length", "num_digits", "num_special", "num_subdomains", "has_ip",
               "domain_length", "digits_ratio"]]


# ===============================================================
# function dataset_preprocess()
# clean the dataset before it can be used to train the model
# ===============================================================

def dataset_preprocess(input_csv="Phishing_URL_Dataset.csv", output_csv="Preprocessed_Dataset.csv"):
    # Load the data from CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Drop duplicate URLs 
    df = df.drop_duplicates(subset="URL")

    # Drop rows missing URL or label values
    df = df.dropna(subset=["URL", "label"])

    # Ensure labels are integers (0 = malicious, 1 =s afe)
    df["label"] = df["label"].astype(int)

    # Save the cleaned DataFrame to a CSV 
    df.to_csv(output_csv, index=False)

    # Return the output filename 
    return output_csv

# ===============================================================
# function trainModel() 
# used to train the Naive Bayes model 
# and print the evaluation metrics of said model
# ===============================================================

def trainModel(csv_file="Preprocessed_Dataset.csv"):
    # Initialize the evaluation metric variables
    accuracy_number = None
    train_time = None
    run_time = None

    # Record when the function begins to run
    func_start = time.time()

    # Load the data
    df = pd.read_csv("Phishing_URL_Dataset.csv")

    # Look at first few rows to see how the data looks
    # Comment out once not needed
    # print(df.head())

    # Split data into the train and test sets
    # Input features is just the url of the given site
    x_data = df["URL"]
    # Target variable is 0 or 1, which indicates whether the URL is malicious or not
    y_data = df["label"]

    # Use train_test_split to split the data, we will have an 80% train and 20% test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data,
        test_size=0.2,              # 80/20 split
        random_state=42,            # Randomize the data
        stratify=y_data,            # Keep the class balance between train and test split
        )


    # Transform your URLs into numbers for ML model to understand
    # High weight to rare but important character patterns, and low weight to common patterns
    # analyzer="char_wb": Look at character sequences
    # ngram_range=(2,4): Use 2-character, 3-character, and 4-character sequences
    # min_df=5: Use patterns that appear in 5 or more URLs
    count_vector = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=5)

    # Fit the tfidf only on the training URLs and transform both train and test
    x_train_vec = count_vector.fit_transform(x_train)
    x_test_vec = count_vector.transform(x_test)

    # Extract numeric features for train and test
    train_extra = extract_url_features(pd.DataFrame({"URL": x_train}))
    test_extra = extract_url_features(pd.DataFrame({"URL": x_test}))

    # Stack text features with numeric features
    x_train_final = hstack([x_train_vec, train_extra])
    x_test_final = hstack([x_test_vec, test_extra])

    # Compute class weights to handle imbalance 
    weights_array = compute_class_weight(class_weight="balanced",
                                        classes=np.array([0, 1]),
                                        y=y_train)

    # Convert weights array to the dictionary for Logistic Regression
    class_weights = {0: weights_array[0], 1: weights_array[1]}

    # Build a Logistic Regression model 
    MLmodel = LogisticRegression(
        max_iter=4500,         # Stop at 4500 iterations
        solver="saga",         # Supports sparse input 
        n_jobs=-1,             # Use all CPU cores
        class_weight=class_weights,
        C=1.5                  # Inverse regularization strength 
    )
    
    # Record the time when the model training starts
    model_start = time.time()
    # Train the ML model
    MLmodel.fit(x_train_final, y_train)
    # Record time when model training ends
    model_end = time.time()

    # Calculate and print the training time
    train_time = model_end - model_start
    print(f"Model took {train_time} seconds to train")

    # Calculate model accuracy
    accuracy_number = MLmodel.score(x_test_final, y_test)
    print(f"Accuracy of model is {accuracy_number}")

    # Generate and print a classification report 
    y_pred = MLmodel.predict(x_test_final)
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Malicious", "Not Malicious"]))

    # Get the runtime of the entire function
    func_end = time.time()
    func_totaltime = func_end - func_start
    print(f"Program runtime is {func_totaltime} seconds")

    # Save trained model, vectorizer, and metrics
    joblib.dump(MLmodel, "trained_model.joblib")
    joblib.dump(count_vector, "vectorizer.joblib")
    joblib.dump(train_time, "train_time.joblib")
    joblib.dump(accuracy_number, "accuracy.joblib")
    joblib.dump(classification_report(y_test, y_pred, target_names=["Malicious", "Not Malicious"]), "class_report.joblib")

# ===============================================================
# function URLpredict()
# uses trained model and vectorizer to predict if
# the URL input is likely to be malicious or not
# ===============================================================

def URLpredict(url):

    # Load the model, vectorizer, and eval metrics
    MLmodel = joblib.load("trained_model.joblib")
    count_vector = joblib.load("vectorizer.joblib")
    loaded_training_time = joblib.load("train_time.joblib")
    loaded_accuracy = joblib.load("accuracy.joblib")
    class_report = joblib.load("class_report.joblib")

    accuracy = round(loaded_accuracy, 2)
    training_time = round(loaded_training_time, 2)
    # Convert URL to numeric features
    url_vec = count_vector.transform([url])
    extra_features = extract_url_features(pd.DataFrame({"URL": [url]}))

    # Combine tfidf and numeric features
    final_features = hstack([url_vec, extra_features])

    # Make prediction using the model
    prediction = MLmodel.predict(final_features)[0]

    if prediction == 0:
        prediction = "URL HAS A HIGH CHANCE OF BEING MALICIOUS"

    elif prediction == 1:
        prediction = "URL HAS A HIGH CHANCE OF BEING NOT MALICIOUS"

    return accuracy, training_time, class_report, prediction

def main():
    #dataset_preprocess()                          
    #trainModel()                                  
    URLpredict("http://www.example.com/login")
    URLpredict("http://www.secure-login-bank.com/verify?id=12345")
    URLpredict("http://www.facebook.com")
    URLpredict("http://www.google.com")
    URLpredict("http://www.amazon.com")
    URLpredict("https://www.rewildingargentina.org")
    URLpredict("http://www.teramill.com")
    URLpredict("http://paypa1-secure-login.com/verify/account")
    URLpredict("http://g00gle-update.net/login")

if __name__ == "__main__":
    main()