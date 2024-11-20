The repository contains code for custom built aspect based sentiment analysis model. Here are the few steps to build the pipeline.

# Downloads
1. Download Stanford POS Tagger from https://nlp.stanford.edu/software/tagger.shtml (update the path around line 34 in main.py)
2. Depending on version (https://chromedriver.chromium.org/downloads) of your Google Chrome, download the chrome driver (update path around line 144 in main.py)

# Build TextBlob
TextBlob needs to be cloned from GitHub (https://github.com/sloria/TextBlob) and the file named "en-sentiment.xml" needs to replaced with TextBlob/textblob/en/en-sentiment.xml and then Textblob needs to be built using-

```python
python setup.py install
```

# Dependencies
Install the dependencies using `pip install -r requirements.txt`

Now that we have all requirements, we can proceed to running our module. One can get the results using the following code - 

```python
from main import get_aspect_reviews
get_aspect_reviews(URL, datetime)
```
In the above code, URL and datetime is to provided for shop's URL and the time (in months) for which reviews need to be extracted. This function would output two pie charts regarding polarity and aspect category distributions and would save a json file which has the sentences categorized for the shop reviews.
