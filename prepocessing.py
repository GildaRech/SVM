import pandas as pd
import numpy as np
import re

exclude='!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~'
def clean(text):
    text=text.lower()
    pattern = re.compile('<.*?>')
    text=pattern.sub(r'', text)
    pattern = re.compile(r'https?://\S+|www\.\S+')
    text=pattern.sub(r'', text)
    for char in exclude:
        text = text.replace(char,' ')
    return text

def prepocessing(dataset):
    df=dataset.sample(20000,ignore_index=True)
    df['review'] = df['review'].apply(clean)
    df['sentiment']=np.where(df['sentiment']=='positive',1,-1)
    return df