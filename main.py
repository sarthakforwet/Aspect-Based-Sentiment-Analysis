## Base
import json
import time
import pickle
import re, os
import numpy as np
import pandas as pd

## Scraping
from bs4 import BeautifulSoup
from selenium import webdriver

# Text Handling
import nltk
from textblob import TextBlob
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tag.stanford import StanfordPOSTagger as POS_Tag

#To tag using stanford pos tagger
nltk.download('punkt')
nltk.download("wordnet")
nltk.download('omw-1.4')

## Text Visualization
import matplotlib.pyplot as plt

# Translator.
from googletrans import Translator
trans = Translator()

lemma = nltk.wordnet.WordNetLemmatizer()
stop_words = stopwords.words("english")
with open("stopwords.txt", "r") as f:
    data = f.read()
    data = data.strip()
    data = data.split("\n")
stop_words.extend(data)

# Removing negation words from stop words.
for _ in range(2):
    for i, e in enumerate(stop_words):
        if e[0]=="n":
            del stop_words[i]

home = r'./stanford-postagger-full'
_path_to_model = home + '/models/english-bidirectional-distsim.tagger'
_path_to_jar = home + '/stanford-postagger.jar'
stanford_tag = POS_Tag(model_filename=_path_to_model, path_to_jar=_path_to_jar)

# Contrast words are included so that sentence with multiple aspects can be segregated.
contrast_words = ["on the contrary", "yet", "but", "still", "rather", "nor", "conversely", "at the same time",
"however", "nevertheless", "despite", "otherwise", "by contrast", "whereas", "unlike", "although", "in contrast",
"notwithstanding", "in spite of","alternatively", "despite this", "in contrast to", "in contrast with"]

# Loading the SVM model.
svm = pickle.load(open("svm_model_v3.pkl", "rb"))
lbl = pickle.load(open("label_encoded_v3.pkl", "rb"))
print("Loaded model utils...")

def get_aspect_reviews(url, datetime=-1):
    """
    Function to run the complete module and get the final sentiments with pie charts
    from just providing the url of the shop on Maps.
    """
    obj = SentimentAnalysis()
    obj.collectData(url, datetime)
    obj.processReviewDf()
    obj.getSentiments()

    # Plot positive and negative distribution.
    data = obj.outDf[obj.outDf["polarity"]!=0]
    pos_reviews_dist = len(data[data["polarity"]>0])/len(data)
    neg_reviews_dist = len(data[data["polarity"]<0])/len(data)
    obj.plot_pie([pos_reviews_dist, neg_reviews_dist], ["Positive", "Negative"])

    obj.classify_aspects()
    
    print("Categorizing reviews...")
    obj.merge_aspSen()
    
    shopName = url.split('/')[5].replace("+", " ") # Get the shopName.
    
    if not os.path.isdir("Segment Reviews"):
        os.mkdir("Segment Reviews")
    
    json.dump(obj.catSep, open(f"Segment Reviews/{shopName}.json", "w"))
    print(f"Saved categorized reviews in {shopName}.json...")
    
    return obj

class SentimentAnalysis:
    def remove_emoji(self, string):
        """
        Function to remove emoji from the sentences.
        """
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

        return emoji_pattern.sub(r'', string)

    def get_review_summary(self, reviews):
        """
        Structure the reviews collected from Google Reviews and convert time into months.
        """

        print("Structuring Reviews...")
        rev_dict = {'Review Rate': [],
            'Review Time': [],
            'Review Text' : []}

        for result in reviews:
            review_rate = result.find('span', class_='ODSEW-ShBeI-H1e3jb')["aria-label"]
            review_time = result.find('span',class_='ODSEW-ShBeI-RgZmSc-date').text
            review_text = result.find('span',class_='ODSEW-ShBeI-text').text
            rev_dict['Review Rate'].append(review_rate)
            rev_dict['Review Time'].append(review_time)
            rev_dict['Review Text'].append(review_text)

        reviewDf = pd.DataFrame(rev_dict)

        reviewDf = reviewDf[reviewDf["Review Text"]!=""]

        # Process Review Time. Converting all in days
        dates = []
        for t in reviewDf["Review Time"].values:
            a, b, _ = t.split(" ")
            if a=="a":
                a = 1
            else:
                a = int(a)

            if b == "year" or b == "years":
                b = 12
            elif b == "month" or b == "months":
                b = 1
            else:
                b = 0

            dates.append(int(a*b))

        reviewDf["monthsAgo"] = dates
        return reviewDf

    def collectData(self, url, dateTime=-1):
        """
        Extract reviews from the provided Url. The url must be of Google Maps.
        dateTime: representing number of months.
        """
        print("Collecting Data...")
        driver = webdriver.Chrome("../ChromeDriver/chromedriver_linux64/chromedriver")
        driver.get(url)

        totRevButton = driver.find_element_by_class_name("Yr7JMd-pane-hSRGPd")
        total_number_of_reviews = totRevButton.text.split()[0]
        #total_number_of_reviews = driver.find_element_by_xpath('//*[@id="pane"]/div/div[1]/div/div/div[2]/div[2]/div/div[2]/div[2]').text.split(" ")[0]
        total_number_of_reviews = int(total_number_of_reviews.replace(',','')) if ',' in total_number_of_reviews else int(total_number_of_reviews)

        totRevButton.click()
        #Find scroll layout
        time.sleep(2)

        # Sorting the reviews on the basis on newly availables.
        sort_button = driver.find_element_by_xpath('/html/body/div[3]/div[9]/div[8]/div/div[1]/div/div/div[2]/div[7]/div[2]/button/span/span')
        sort_button.click()
        time.sleep(1)
        newest_button = driver.find_element_by_xpath('/html/body/div[3]/div[3]/div[1]/ul/li[2]')
        action = webdriver.common.action_chains.ActionChains(driver)
        action.move_to_element_with_offset(newest_button, 5, 5)
        action.click()
        action.perform()

        time.sleep(2)
        
        # Get scrollable section.
        scrollable_div = driver.find_element_by_xpath('//*[@id="pane"]/div/div[1]/div/div/div[2]')

        #Scroll as many times as necessary to load almost all reviews
        for i in range(0,(round(total_number_of_reviews/10 - 1))):
                driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight',
                        scrollable_div)

                response = BeautifulSoup(driver.page_source, 'html.parser')
                reviews = response.find_all('div', class_='ODSEW-ShBeI NIyLF-haAclf gm2-body-2')
                review_time = reviews[-1].find('span',class_='ODSEW-ShBeI-RgZmSc-date').text

                # Get date in terms of month numbers.
                a, b, _ = review_time.split(" ")

                if a=="a":
                    a = 1
                else:
                    a = int(a)

                if b == "year" or b == "years":
                    b = 12
                elif b == "month" or b == "months":
                    b = 1
                else:
                    b = 0

                if int(a*b) > dateTime:
                    break

                time.sleep(1)

        # Clicking 'More' Button(s).
        btns = driver.find_elements_by_xpath("//jsl/button")
        for e in btns:
            e.click()

        response = BeautifulSoup(driver.page_source, 'html.parser')
        reviews = response.find_all('div', class_='ODSEW-ShBeI NIyLF-haAclf gm2-body-2')

        self.reviewDf = self.get_review_summary(reviews)

        # If we are specifying dateTime other than -1 extract only till those months.
        if dateTime!= -1:
            self.reviewDf = self.reviewDf[self.reviewDf["monthsAgo"]<=dateTime]

        driver.close()

    def processReviewDf(self):
        """
        Function to process the reviews to remove predefined noise from them.
        """
        print("Processing Reviews to remove noise...")
        review_clean = []
        for review in self.reviewDf["Review Text"].values:
            if "(Original)" in review:
                review, original = review.split("(Original)")

            if "(Translated by Google)" in review:
                review = review.replace("(Translated by Google)", "")
                review = review.strip()

            review_clean.append(review)

        self.reviewDf["review_clean"] = review_clean

    def posTag(self, review):
        """
        Function to tag words in a review to their respective tags.
        """
        return stanford_tag.tag(word_tokenize(review))

    def getAspects(self, sen):
        """
        Function to extract aspects from a sentence.
        Currently Noun(s) are considered as aspect since most approaches
        consider the same.
        """
        rows = []
        aspect_tags = ["NN", "NNS", "NNP", "NNPS"]
        target = []
        tagged_sen = self.posTag(sen)
        for word, token in tagged_sen:
            if token in aspect_tags:
                target.append(word)
                
        return ", ".join([e for e in target])

    def getSentiments(self):
        """
        The driver function which is used to preprocess the reviews and extract sentiments
        from the formed sentences.
        """
        print("Getting Sentiments...")
        sentences = []
        for row in self.reviewDf.iterrows():
            row = row[1]
            a = row["review_clean"].split(". ")
            for e in a:
                e = e.strip()
                e = self.remove_emoji(e) # Removing Emoji
                e = e.strip()
                if len(e.split(" "))<3:
                    continue

                sentences.append(e)

        polarity = []
        #subjectivity = [] Not using subjectivity as of now.
        sens = []
        aspects = []
        self.outDf = pd.DataFrame()
        for i, sen in enumerate(sentences):                
            flag = 0
            sen = sen.lower()
            for e in contrast_words:
                x = []
                if len(e.split(" "))==1:
                    senList = sen.split(" ")
                    if e in senList:
                        flag = 1
                        x = [" ".join([i for i in senList[:senList.index(e)]]), " ".join([i for i in senList[senList.index(e)+1:]])]

                else:
                    if e in sen:
                        flag = 1
                        x = sen.split(e)

                for each in x:
                    each = each.strip()
                    if len(each.split(" "))<3:
                        continue

                    # Translating each.
                    each = trans.translate(each).text

                    # Performing Stop Word Removal.
                    tokenizeSen = word_tokenize(each)
                    stopSen = [wrd for wrd in tokenizeSen if not wrd.lower() in stop_words]

                    ## Lemmatizing the sentence.
                    lemSen = " ".join([lemma.lemmatize(wrd, "v") for wrd in stopSen])

                    asp = self.getAspects(lemSen)
                    aspects.append(asp)
                    blob = TextBlob(lemSen)
                    sentiment = blob.sentiment
                    polarity.append(sentiment.polarity)
                    #subjectivity.append(sentiment.subjectivity)
                    sens.append(each)

            if not flag:
                # Translate
                sen = trans.translate(sen).text

                # Performing Stop Word Removal.
                tokenizeSen = word_tokenize(sen)
                stopSen = [wrd for wrd in tokenizeSen if not wrd.lower() in stop_words]

                ## Lemmatizing the sentence.
                lemSen = " ".join([lemma.lemmatize(wrd, "v") for wrd in stopSen])

                asp = self.getAspects(lemSen)
                aspects.append(asp)
                blob = TextBlob(lemSen)
                sentiment = blob.sentiment
                polarity.append(sentiment.polarity)
                #subjectivity.append(sentiment.subjectivity)
                sens.append(sen)

        self.outDf["sentence"] = sens
        self.outDf["polarity"] = polarity
        #self.outDf["subjectivity"] = subjectivity
        self.outDf["aspects"] = aspects

        print(f"Length of collected data: {len(self.reviewDf)}")
        print(f"Length of final data: {len(self.outDf)}")

    def plot_pie(self, x, labels=None):
        """
        Function to plot pie chart of the given data and labels.
        """
        plt.figure(figsize=(6,6))

        if labels:
            plt.pie(x, autopct="%.2f%%", labels=labels)
        else:
            plt.pie(x, autopct="%.2f%%")
        plt.show()

    def classify_aspects(self):
        """
        Classify Aspects into predefined categories using the trained
        SVM model.
        """
        # Extracting the negative aspects.
        negRev = self.outDf[self.outDf["polarity"]<0]
        negRev.dropna(inplace=True)
        asp = []
        for row in negRev.iterrows():
            row = row[1]
            asp.extend(row["aspects"].split(", "))

        asp = list(set(asp))

        # Loading and assigning word embeddings.
        glove_embeddings = {}

        with open('glove.6B/glove.6B.50d.txt', 'r') as fopen:
            for line in fopen:
                split_line = line.split()
                word = split_line[0]
                emb = np.array(split_line[1:], dtype=np.float64)
                glove_embeddings[word] = emb

        word_embeddings = {}
        for word in asp:
            if word in glove_embeddings.keys():
                word_embeddings[word] = glove_embeddings[word]
            else:
                continue

        wordEmbDf = pd.DataFrame({"word":word_embeddings.keys(), "embedding":word_embeddings.values()})

        predCategory = []
        for row in wordEmbDf.iterrows():
            row = row[1]
            pred = lbl.inverse_transform(svm.predict(np.expand_dims(row["embedding"], axis=0)))
            predCategory.append(pred[0])

        wordEmbDf["predCategory"] = predCategory

        portions = []
        labels = []
        self.aspCat = {}
        for cat in wordEmbDf.predCategory.unique():
            tmp = wordEmbDf[wordEmbDf["predCategory"]== cat]
            self.aspCat[cat] = tmp.word.values
            portions.append(round(len(tmp)/len(wordEmbDf)*100, 2))
            labels.append(cat)

        self.plot_pie(portions, labels)

    
    def merge_aspSen(self):
        self.catSep = {}
        catBool = {}

        for cat in self.aspCat.keys():
            self.catSep[cat] = []
            catBool[cat] = 0

        negDf = self.outDf[self.outDf["polarity"]<0]

        for row in negDf.iterrows():
            row = row[1]

            if type(row["aspects"]) == float:
                    continue

            aspects = row["aspects"].split(", ")

            for asp in aspects:
                for cat in self.aspCat.keys():
                    if catBool[cat]:
                        continue

                    if asp in self.aspCat[cat]:
                        self.catSep[cat].append(row["sentence"])
                        catBool[cat] = 1
                        break

            # Resetting the category.
            for k in catBool:
                catBool[k] = 0
        