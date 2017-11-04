import nltk
from nltk.corpus import stopwords

s = '''We're looking for a mobile designer for our Yelp-like startup (but cooler!). Designer needs the following experience:

- Mobile UI/UX experience
- Uses Sketch/other design software
- Knows Zeplin/other asset software
- Uses Yelp

IMPORTANT: Must provide examples of UI/UX development work, particularly in this space.'''

special = ['~', '`', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '-', '+', '=', '{', '[', '}', ']', '|', '\\', ':', ';', '\'', '"', ',', "<", '.', '>', '/', '?']

def compress(text):
    # remove special characters
    for sp in special:
        text = text.replace(sp, ' ')

    # split into words
    words = text.split()

    # make words lower
    words = [w.lower() for w in words]
    
    # remove stop words
    stop = stopwords.words('english')
    result = [w for w in words if w not in stop]
    print(result)

compress(s)
