import os
import numpy as np
import re
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils

# Load from ex6data1
# You will have X, y as keys in the dict data
# data = loadmat(os.path.join('Data', 'ex6data1.mat'))
# X, y = data['X'], data['y'][:, 0]

# Plot training data
# utils.plotData(X, y)
# pyplot.show()

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
# C = 100

# model = utils.svmTrain(X, y, C, utils.linearKernel, 1e-3, 20)
# utils.visualizeBoundaryLinear(X, y, model)
# pyplot.show()

def gaussianKernel(x1, x2, sigma):
    sim = 0
    gaussian_1 = ((x1-x2)**2) / (2*(sigma**2))
    gaussian_2 = np.sum(gaussian_1)
    sim = np.exp(-gaussian_2)
    return sim

# x1 = np.array([1, 2, 1])
# x2 = np.array([0, 4, -1])
# sigma = 2

# sim = gaussianKernel(x1, x2, sigma)

# print('Gaussian Kernel between x1 = [1, 2, 1], x2 = [0, 4, -1], sigma = %0.2f:'
#       '\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n' % (sigma, sim))

# Load from ex6data2
# You will have X, y as keys in the dict data
# data = loadmat(os.path.join('Data', 'ex6data2.mat'))
# X, y = data['X'], data['y'][:, 0]

# Plot training data
# utils.plotData(X, y)
# pyplot.show()

# SVM Parameters
# C = 1
# sigma = 0.1

# model= utils.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
# utils.visualizeBoundary(X, y, model)
# pyplot.show()

# Load from ex6data3
# You will have X, y, Xval, yval as keys in the dict data
# data = loadmat(os.path.join('Data', 'ex6data3.mat'))
# X, y, Xval, yval = data['X'], data['y'][:, 0], data['Xval'], data['yval'][:, 0]
# print(X.shape) 
# print(y.shape) # print(X[0])

# model = utils.svmTrain(X, y, 3, gaussianKernel, args=(0.1,))
# aaa = model['alphas'] print(aaa.shape) # print(model)

# Plot training data
# utils.plotData(X, y)
# pyplot.show()


def dataset3Params(X, y, Xval, yval):
    # You need to return the following variables correctly.
    # C = 1
    # sigma = 0.3

    number_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    predicts = []

    for i in range(len(number_vec)):
        for j in range (len(number_vec)):
            C = number_vec[i]
            sigma = number_vec[j]
            model= utils.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
            predictions = utils.svmPredict(model, Xval)
            err_cv = np.mean(predictions != yval)
            temp_tuple = (C, sigma, err_cv)
            predicts.append(temp_tuple)

    # print(len(predicts))
    tam = sorted(predicts, key=lambda tup: tup[2])
    # print(tam[0])
    # return C, sigma
    return tam[0][0], tam[0][1]

# Try different SVM Parameters here
# C, sigma = dataset3Params(X, y, Xval, yval)

# # Train the SVM
# model = utils.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
# utils.visualizeBoundary(X, y, model)
# print(C, sigma)
# pyplot.show()

# model = utils.svmTrain(X, y, 1, gaussianKernel, args=(0.1,))
# utils.visualizeBoundary(X, y, model)
# pyplot.show()

def processEmail(email_contents, verbose=True):
    """
    Preprocesses the body of an email and returns a list of indices 
    of the words contained in the email.    
    
    Parameters
    ----------
    email_contents : str
        A string containing one email. 
    
    verbose : bool
        If True, print the resulting email after processing.
    
    Returns
    -------
    word_indices : list
        A list of integers containing the index of each word in the 
        email which is also present in the vocabulary.
    
    Instructions
    ------------
    Fill in this function to add the index of word to word_indices 
    if it is in the vocabulary. At this point of the code, you have 
    a stemmed word from the email in the variable word.
    You should look up word in the vocabulary list (vocabList). 
    If a match exists, you should add the index of the word to the word_indices
    list. Concretely, if word = 'action', then you should
    look up the vocabulary list to find where in vocabList
    'action' appears. For example, if vocabList[18] =
    'action', then, you should add 18 to the word_indices 
    vector (e.g., word_indices.append(18)).
    
    Notes
    -----
    - vocabList[idx] returns a the word with index idx in the vocabulary list.
    
    - vocabList.index(word) return index of word `word` in the vocabulary list.
      (A ValueError exception is raised if the word does not exist.)
    """
    # Load Vocabulary
    vocabList = utils.getVocabList()

    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================
    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers
    # hdrstart = email_contents.find(chr(10) + chr(10))
    # email_contents = email_contents[hdrstart:]

    # Lower case
    email_contents = email_contents.lower()
    
    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.compile('<[^<>]+>').sub(' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.compile('[0-9]+').sub(' number ', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.compile('(http|https)://[^\s]*').sub(' httpaddr ', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.compile('[^\s]+@[^\s]+').sub(' emailaddr ', email_contents)
    
    # Handle $ sign
    email_contents = re.compile('[$]+').sub(' dollar ', email_contents)
    
    # get rid of any punctuation
    email_contents = re.split('[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', email_contents)

    # remove any empty word string
    email_contents = [word for word in email_contents if len(word) > 0]
    
    # Stem the email contents word by word
    stemmer = utils.PorterStemmer()
    processed_email = []
    # print(email_contents) print(len(vocabList))
    for word in email_contents:
        # Remove any remaining non alphanumeric characters in word
        word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()
        word = stemmer.stem(word)
        processed_email.append(word)

        if len(word) < 1:
            continue
        
        try:
            word_indices.append(vocabList.index(word) + 1)
        except ValueError:
            pass 

    if verbose:
        print('----------------')
        print('Processed email:')
        print('----------------')
        print(' '.join(processed_email))
    return word_indices

#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.

# Extract Features
# with open(os.path.join('Data', 'emailSample1.txt')) as fid:
#     file_contents = fid.read()

# word_indices  = processEmail(file_contents)

# #Print Stats
# print('-------------')
# print('Word Indices:')
# print('-------------')
# print(word_indices)

# vocabList = utils.getVocabList()
# sss = vocabList.index("aa")
# print(sss)

def emailFeatures(word_indices):
    features_vector = np.zeros(1899)
    features_vector[[x - 1 for x in word_indices]] = 1
    return features_vector




data = loadmat(os.path.join('Data', 'spamTrain.mat'))
X = data["X"]
y = data["y"]

y = y.flatten()
print(y[0])
tam = X[0]
print(tam)



# sss = X[:5] sss_y = y[:5] print(sss.shape) 


# Train the SVM
model = utils.svmTrain(X, y, 3, gaussianKernel, args=(0.1,))
# model = utils.svmTrain(X, y, 3, utils.linearKernel, args=(0.01,))
# predictions = utils.svmPredict(model, X)
# err_train = np.mean(predictions == y) * 100
# print('Train Accuracy: {} % '.format(err_train))

# data = loadmat(os.path.join('Data', 'spamTest.mat'))
# Xtest = data["Xtest"]
# ytest = data["ytest"]

# ytest = ytest.flatten()

# predictions_test = utils.svmPredict(model, Xtest)
# err_test = np.mean(predictions_test == ytest) * 100
# print('Test Accuracy: {} % '.format(err_test))



## ================= Part 5: Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
#  weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.
#


## =================== Part 6: Try Your Own Emails =====================
#  Now that you've trained the spam classifier, you can use it on your own
#  emails! In the starter code, we have included spamSample1.txt,
#  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
#  The following code reads in one of these emails and then uses your 
#  learned SVM classifier to determine whether the email is Spam or 
#  Not Spam

# filenames = ['emailSample2.txt', 'emailSample1.txt', 'spamSample2.txt', 'spamSample1.txt']

# for filename in filenames:
#     with open(os.path.join('Data', filename)) as fid: file_contents = fid.read()
#     word_indices  = processEmail(file_contents, False)
#     x = emailFeatures(word_indices)
#     # print(x.shape)

#     prediction = utils.svmPredict(model, x)
#     print('\nProcessed {}\n\nSpam Classification: {}\n'.format(filename, prediction))
#     print('(1 indicates spam, 0 indicates not spam)\n\n')

prediction = utils.svmPredict(model, tam)
print('\nProcessed {}\n\nSpam Classification: {}\n'.format("lam test", prediction))
print('(1 indicates spam, 0 indicates not spam)\n\n')