"""
Build a sentiment classifier to classify restaurant reviews.
"""

# import all necessary libraries
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import lightgbm as lgb
import warnings
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# df = pd.read_csv('reviews.csv', delimiter='\t')
# alternatively, you can read the data directly from my Github without worrying about the file path
df = pd.read_csv(
    'https://raw.githubusercontent.com/JinL-Zhang/2023_Fall_Deep_Learning_and_NLP/main/Sentiment_Classification/reviews.csv', delimiter='\t')
df

"""Pre-processing

The target label 'RatingValue' will be binned into negative (ratings 1 & 2),
neutral (rating 3), and positive (ratings 4 & 5) sentiment.
The binned ratings will be encoded with negative as 0, neutral as 1,
and positive as 2. This new column will be called Sentiment
"""

df['Sentiment'] = pd.cut(df['RatingValue'], bins=[0, 2, 3, 5], labels=[
                         0, 1, 2], include_lowest=True)
df['Sentiment']

"""
Since the ratings are very unbalanced (too many more positive (2) ratings than
negative (0) ratings), we will drop positive ratings in order to balance the data
so that we have approximately equal numbers of negative, neutral and positive ratings.
To achieve this, we can use undersampling method provided in the imblearn library
"""

# sampling_strategy='auto' indicates that it will resample all classes but not
# the minority class. This will result in 3 classes of equal number of instances
# based on the number of instances of the minority class
# (which is the class with the least number of instances)
under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=12)
X_resampled, y_resampled = under_sampler.fit_resample(
    df.drop('Sentiment', axis=1), df['Sentiment'])

"""
It is noteworthy that after undersampling to ensure equal instances in each class,
only 474 out of original 1920 records are retained
"""
# Forming the dataframe with the desired format
X_resampled['Number'] = range(1, X_resampled.shape[0] + 1)
df_formatted = pd.concat(
    [X_resampled['Number'], y_resampled, X_resampled[['Review']]], axis=1)

"""
Now, we can split the data into training and validation sets and save them as train.csv and valid.csv
The validation data will be used for model selection and evaluation.
"""
# 20% of the data are used for validation as the dataset is small after undersampling,
# leading more training data to be needed for a better model performance
df_train, df_valid = train_test_split(
    df_formatted, test_size=0.2, random_state=42)

# Save to CSV
df_train.to_csv('train.csv', index=False)
df_valid.to_csv('valid.csv', index=False)

"""
Now, we can loads the training set from train.csv and start to trains the
TfidfVectorizer (for BoW representation) and the ML classifiers
"""

df_train = pd.read_csv('train.csv')

'''
Train the TfidfVectorizer on the training set only (not including the validation set).
Note: tokenizing and filtering of stopwords are all included in TfidfVectorizer,
which builds a dictionary of features and transforms documents to feature vectors (BoW representation).

Also, to further enrich the BoW representation to enable the model to potentially capture more
variances and contexts (sequential orderings) within the text data, we will not use the simple term-frequency BoW
representation, but the tf-idf representations with n-grams incorporated
(which is the reason why I used TfidfVectorizer, rather than the simpler CountVectorizer)

In this case, since the training set is small (number of observations < 500),
we can use a bigger n for the n-grams without worrying dimensionality explosions
'''

X_train = df_train['Review']
y_train = df_train['Sentiment']
tfidf_vect = TfidfVectorizer(ngram_range=(1, 5),  # all n-grams from the unigram to the 5-gram are extracted
                             max_df=.9,  # ignore the terms that have a document frequency strictly higher than 90%
                             # This can also detect and filter corpus-specific stop words based on intra corpus document frequency of terms.
                             min_df=.05)  # ignore the terms that have a document frequency strictly lower than 5%
# This can remove the infrequent words (noises) and reduces the chance of overfitting
# (and potentially reduces the sparsity of the feature space)
'''
Fun fact: the term with the highest frequency is 'it' even after setting the max_df=.8 and min_df=.1 threshold
'''
X_train_counts = tfidf_vect.fit_transform(X_train)


# define the custom evaluation function
def evaluate_model(y_true, y_pred):
    """
    This function is like a standard sklearn evaluation function that accepts y_pred
    and y_true as the only parameters, and will print out all the performance metrics
    required in the desired format
    """

    # Calculate the overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # the averaged F1-score uses 'macro'
    # which calculates metrics for each label, and find their unweighted mean.
    average_f1 = f1_score(y_true, y_pred, average='macro')

    # Since the target variable is multiclass/multilabel, we will set 'average'
    # to be 'None', so that the f1 scores for each class is returned
    # NOte: the index of the returned f1 score list correspond to the numeric class label
    # so class_f1[0] correspond to the f1 score for class label 0 (negative review)
    class_f1 = f1_score(y_true, y_pred, average=None)

    # Get the confusion matrix, with normalize='true', we will compute a
    # normalized confusion matrix where each row (ground truth) is normalized
    # to have a sum of 1.0.
    conf_matrix = confusion_matrix(y_true, y_pred, normalize='true')

    # Define the class labels in strings
    classes = ['negative', 'neutral', 'positive']

    # Print the results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nAverage F1 Score: {average_f1:.4f}")
    print("\nClass-wise F1 Scores:")
    for i, c in enumerate(classes):
        print(f"  {c}: {class_f1[i]:.4f}")

    print("\nConfusion Matrix:")
    conf_matrix_df = pd.DataFrame(conf_matrix, index=classes, columns=classes)
    print(conf_matrix_df)


'''
Now, we can train a classifier to try to predict the rating class of a reivew.
Let’s start with a naïve Bayes classifier, which provides a nice baseline for this task.
We will switch to other models if the performance is not satisfactory
'''

print('\n************************************************************')
print('-------------------------------------------------------------')
print(" TTTTT  RRRRR    AAAA  IIIII  N    N  IIIII  N    N  GGGGG ")
print("   T    R    R  A    A   I    NN   N    I    NN   N G      ")
print("   T    RRRRR   AAAAAA   I    N N  N    I    N N  N G  GGG  ")
print("   T    R   R   A    A   I    N  N N    I    N  N N G    G  ")
print("   T    R    R  A    A IIIII  N   NN  IIIII  N   NN  GGGGG  ")
print('\n************************************************************')
print('-------------------------------------------------------------')

nb_clf = MultinomialNB(alpha=.5).fit(X_train_counts, y_train)
y_pred_nb = nb_clf.predict(X_train_counts)
print('     Training Performance Evaluation of Naive Bayes    \n')
evaluate_model(y_train, y_pred_nb)


# try the linear svm
svm = SGDClassifier(loss='hinge', penalty='elasticnet',
                    alpha=.001, random_state=42,
                    tol=None)
svm.fit(X_train_counts, y_train)
y_pred_svm = svm.predict(X_train_counts)
print('\n**********************************************************')
print('-------------------------------------------------------')
print('       Training Performance Evaluation of SVM    \n')
evaluate_model(y_train, y_pred_svm)


# try the Gradient Boosting Models (LightGBM)
# Set hyperparameters
params = {
    'objective': 'multiclass',
    'num_class': 3,  # Number of classes
    'metric': 'multi_logloss',
    # 'num_leaves':20,
    # 'max_depth': 7,
    # 'n_estimators': 30,
}

# Create a LightGBM classifier, set 'verbose' to -1 to avoid warning reporting
lgb_model = lgb.LGBMClassifier(**params, verbose=-1)

# set X_train_counts to np.float32 as required by the lightGBM model
lgb_model.fit(X_train_counts.astype(np.float32), y_train)
y_pred_lgb = lgb_model.predict(X_train_counts.astype(np.float32))

print('\n**********************************************************')
print('----------------------------------------------------------')
print('       Training Performance Evaluation of LightGBM    \n')
evaluate_model(y_train, y_pred_lgb)

"""
Now, we can evaluate the 3 models on the validation set to select the right model
Note that the TfidfVectorizer should not be fitted on the validation set, but performing
transformation only to simulate unseen data
"""

# load the validation set
df_valid = pd.read_csv('valid.csv')
X_valid = df_valid['Review']
y_valid = df_valid['Sentiment']

# transform validation data into the BoW representation
X_valid_counts = tfidf_vect.transform(X_valid)

# evaluate on validation set
print('\n***********************************************************')
print('------------------------------------------------------------')
print("  TTTTT  EEEEE  SSSSSS  TTTTT  IIIII  N     N  GGGGG")
print("    T    E      S         T      I    NN    N  G")
print("    T    EEEE    SSSS     T      I    N N   N  G  GGG")
print("    T    E           S    T      I    N  N  N  G    G")
print("    T    E           S    T      I    N   N N  G    G")
print("    T    EEEEE  SSSSS     T    IIIII  N    NN   GGGGG")
print('\n**********************************************************')
print('------------------------------------------------------------')
print('  Validation/Testing Performance Evaluation of Naive Bayes \n')
y_pred_nb = nb_clf.predict(X_valid_counts)
evaluate_model(y_valid, y_pred_nb)

print('\n**********************************************************')
print('------------------------------------------------------------')
print('  Validation/Testing Performance Evaluation of SVM   \n')
y_pred_svm = svm.predict(X_valid_counts)
evaluate_model(y_valid, y_pred_svm)

print('\n**********************************************************')
print('------------------------------------------------------------')
print('   Validation/Testing Performance Evaluation of LightGBM   \n')
y_pred_lgb = lgb_model.predict(X_valid_counts.astype(np.float32))
evaluate_model(y_valid, y_pred_lgb)

print('''
According to the training and testing evaluation results,
all the models show some degrees of overfitting, which may be partially attributed to the OOV
(out-of-vocabulary) problem, and their validation performances are not impressive
though all of them perform above chance (0.5), which may suggest that either the BoW
representation is still not informative enough to enable the models to capture the underlying
data patterns, or there is not enough training data due to the undersampling.

In addition, since Naive Bayes outperforms all the other 2 models, I may finally choose the
Naive Bayes (with the hyper-parameter setting I have in the code) to predict on the unseen test set.

''')
