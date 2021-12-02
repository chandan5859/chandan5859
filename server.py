from flask import Flask, request, jsonify, render_template

import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import joblib

### First, let's try LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
# Now, let's try SUPPORT VECTOR MACHINE (SVM)
from sklearn.svm import SVC # "Support Vector Classifier" 
# Now, let's try XGBoost Classifier
from xgboost import XGBClassifier

#nltk.download('stopwords')
#nltk.download('wordnet')


app = Flask(__name__)


def translate_personality(b_Pers, personality):
    # transform mbti to binary vector
    return [b_Pers[l] for l in personality]


def translate_back(b_Pers_list, personality):
    # transform binary vector to mbti personality
    
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s



def pre_process_data(data, remove_stop_words=True, remove_mbti_profiles=True):

    b_Pers = {'I':0, 'E':1, 
          'N':0, 'S':1, 
          'F':0, 'T':1, 
          'J':0, 'P':1}


    b_Pers_list = [{0:'I', 1:'E'}, 
                {0:'N', 1:'S'}, 
                {0:'F', 1:'T'}, 
                {0:'J', 1:'P'}]


    # We want to remove these from the posts
    unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
        'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
    
    unique_type_list = [x.lower() for x in unique_type_list]


    # Lemmatize
    stemmer = PorterStemmer()
    lemmatiser = WordNetLemmatizer()

    # Cache the stop words in local storage for speed 
    cachedStopWords = stopwords.words("english")

    list_personality = []
    list_posts = []
    len_data = len(data)
    i=0
    
    for row in data.iterrows():
        #i+=1
        #if (i % 500 == 0 or i == 1 or i == len_data):
        #    print("%s of %s rows" % (i, len_data))
            #print(row) # refers to the row object in dataframe
            #print(row[1]) # row[1] refers to the data itself, row[0] would give the index

        print(row)

        ##### Remove and clean comments
        posts = row[1].posts
        print(posts)
        
        # Remove URLs
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', str(posts))
        
        # phrases that are not english letters are removed
        temp = re.sub("[^a-zA-Z]", " ", temp)
        temp = re.sub(' +', ' ', temp).lower()
        
        if remove_stop_words:
            # Lemmatize the word -> i.e. "running", "ran", etc. becomes --> "run"
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])
        
        # rare occurrences of the the class types "INTP", "INTJ" etc. also occur in the text, can be removed
        if remove_mbti_profiles:
            for t in unique_type_list:
                temp = temp.replace(t,"")

        print(row[1].type)

        type_labelized = translate_personality(b_Pers, row[1].type)
        list_personality.append(type_labelized)
        list_posts.append(temp)

    list_posts = np.array(list_posts)
    
    list_personality = np.array(list_personality)
    return list_posts, list_personality



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    form_input = str(list(request.form.values())[0])
    #form_input = new_test_posts_1 #### this should be the input from the form

    #cntizer = CountVectorizer(analyzer="word", 
    #                      max_features=1500, 
    #                      tokenizer=None,    
    #                      preprocessor=None, 
    #                      stop_words=None,  
    #                      max_df=0.7,
    #                      min_df=0.1) 

    #tfizer = TfidfTransformer()

    b_Pers = {'I':0, 'E':1, 
          'N':0, 'S':1, 
          'F':0, 'T':1, 
          'J':0, 'P':1}


    b_Pers_list = [{0:'I', 1:'E'}, 
                {0:'N', 1:'S'}, 
                {0:'F', 1:'T'}, 
                {0:'J', 1:'P'}]


    cntizer = pickle.load(open("./models/vectorizer/cntizer.pkl", "rb"))
    tfizer = pickle.load(open("./models/vectorizer/tfizer.pkl", "rb"))

    df_new = pd.DataFrame(columns=('posts', 'type'))
    df_new.loc[0] = [form_input , 'INFP']

    test_data = pd.DataFrame(data={'type': ['INTP'], 'posts': [df_new.posts]})
    print(test_data.posts)
    print(test_data.type)

    test_posts, dummy  = pre_process_data(test_data, remove_stop_words=True)

    print(test_posts)

    result_list = []

    test_cnt = cntizer.transform(test_posts)
    test_tfidf = tfizer.transform(test_cnt).toarray()

    ### FOR LOGISTIC Regression
    for l in range(0, 4):
        
        # make predictions for my test data
        if l==0:
            model = joblib.load('./models/logreg/model1.pkl')
        if l==1:
            model = joblib.load('./models/logreg/model2.pkl')
        if l==2:
            model = joblib.load('./models/logreg/model3.pkl')
        if l==3:
            model = joblib.load('./models/logreg/model4.pkl')
            
        making_test_pred = model.predict(test_tfidf)
        print(making_test_pred)
        result_list.append(making_test_pred[0])
    
    logreg_text = "Logistic Regression:     " + translate_back(b_Pers_list, result_list) # --> actual is INTP
    result_list = []

    ### FOR SVM
    for l in range(0, 4):
        
        # make predictions for my test data
        if l==0:
            model = joblib.load('./models/svm/model1.pkl')
        if l==1:
            model = joblib.load('./models/svm/model2.pkl')
        if l==2:
            model = joblib.load('./models/svm/model3.pkl')
        if l==3:
            model = joblib.load('./models/svm/model4.pkl')

        making_test_pred = model.predict(test_tfidf)
        print(making_test_pred)
        result_list.append(making_test_pred[0])

    svm_text = "SVM:     " + translate_back(b_Pers_list,result_list) # --> actual is INTP
    result_list = []


    ### FOR XGBoost
    for l in range(0, 4):
        
        # make predictions for my test data
        if l==0:
            model = joblib.load('./models/xgb/model1.pkl')
        if l==1:
            model = joblib.load('./models/xgb/model2.pkl')
        if l==2:
            model = joblib.load('./models/xgb/model3.pkl')
        if l==3:
            model = joblib.load('./models/xgb/model4.pkl')

        making_test_pred = model.predict(test_tfidf)
        print(making_test_pred)
        result_list.append(making_test_pred[0])

    xgb_text = "XGBoost:     " + translate_back(b_Pers_list, result_list) # --> actual is INTP
    result_list = []    


    return render_template('index.html', logreg_text = logreg_text, svm_text = svm_text, xgb_text = xgb_text)



if __name__ == "__main__":
    app.run(debug=True)
