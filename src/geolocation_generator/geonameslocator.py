import json
import pandas as pd
from nltk.tokenize import RegexpTokenizer

#load gazetteer
def load_gazetteer(locdictionary, locationdata):
    # Dictionary mapping (lower-case) strings of locations to ids of places
    if locdictionary is None:
        locdictionary = json.load(open('geonameslocator/locdictionary.json'))
    # Data frame mapping ids of places to metadata about these places
    if locationdata is None:
        locationdata = pd.read_csv("geonameslocator/countriesdatap.tsv", sep='\t', low_memory=False, index_col="geonameid")
    
    global tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    
    return locdictionary, locationdata

# Turn tokens into a sequence of n-grams
def word_ngrams(tokens, ngrams):
    exclude = ['province', 'town', 'village', 'hospital', 'sea port', 'port', 'district', \
           'island', 'state', 'municipality', 'governorate', 'axis', 'region', \
           'hub', 'continent', 'idp camp', 'refugee camp', 'sub-district',\
           'subdistrict', 'sub district', 'residential complex', 'basin', 'administrative centre', \
           'administrative center', 'detention centre', 'detention center', 'centre', 'center', \
           'subdivision', 'oil refinery', 'territory', 'lga', 'division', 'country', 'checkpoint', \
           'river', 'market', 'department', 'airport', 'local government area', 'bus station', \
           'military camp', 'highway', 'extension', 'capital', 'border point', 'crossing point', \
           'border', 'road', 'channel', 'mountain', 'prefecture', 'volcano', 'highland', 'lake', \
           'al', 'and', 'sub', 'primary school', 'university city', 'university', 'city', 'school', 'camp', 'area', 'station', \
              'the', 'national', 'park', 'central bank']
    min_n, max_n = 1, ngrams
    if max_n != 1:
        original_tokens = tokens
        tokens = []
        n_original_tokens = len(original_tokens)
        for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
            for i in range(n_original_tokens - n + 1):
                if not " ".join(original_tokens[i: i + n]) in exclude:
                    tokens.append(" ".join(original_tokens[i: i + n]))
    return tokens

# Finds all the longest string matches for a piece of text
def find_names(text, locdictionary = None, locationdata = None):
    locdictionary, locationdata = load_gazetteer(locdictionary, locationdata)
    tokens = word_ngrams(tokenizer.tokenize(text.lower()), 5)
    m = set()
    for token in tokens:
        if token in locdictionary:
            m.add(token)
    # filter out matched places that are substrings of another matched place
    k_list = list(m)
    for i, k in enumerate(k_list):
        for k2 in k_list[:i]:
            if k in k2 and k in m:
                m.remove(k)
        for k2 in k_list[i+1:]:
            if k in k2 and k in m:
                m.remove(k)
    return m

def match_with_gazetteer(ents: list):
    for ent in reversed(range(len(ents))):
        if len(find_names(ents[ent]["ent"])) == 0:
            del ents[ent]
    return ents

