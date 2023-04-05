import spacy
import re
import nltk
from nltk.corpus import stopwords
from .geonameslocator import match_with_gazetteer
#from transformers import AutoTokenizer, AutoModelForTokenClassification
#from transformers import pipeline


#load models for every language. Load once per session
def load_spacy_models():
    global nlp_LANG
    try:
        nlp_LANG
    except NameError:
        nlp_LANG = spacy.load("en_core_web_sm")
        nlp_LANG.add_pipe("language_detector")

    global spacy_FR
    try:
        spacy_FR
    except NameError:
        spacy_FR= spacy.load("fr_core_news_md")

    global spacy_ES
    try:
        spacy_ES
    except NameError:
        spacy_ES= spacy.load("es_core_news_md")
    
    global spacy_EN
    try:
        spacy_EN
    except NameError:
        spacy_EN= spacy.load("en_core_web_md")

def load_transformers_model(model_path):

    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from transformers import pipeline

    global tokenizer
    try:
        tokenizer
    except NameError:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    global trans_model
    try:
        trans_model
    except NameError:
        trans_model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    global trans_nlp
    try:
        trans_nlp
    except NameError:
        trans_nlp = pipeline("ner", model=trans_model, tokenizer=tokenizer)
    
    return trans_nlp

# get data language
def get_language(df):
    load_spacy_models()
    string = ' '.join(df.text)[0:10000]
    language = nlp_LANG(string)._.language

    if language == "fr":
        model = spacy_FR
    elif language == "es":
        model = spacy_ES
    elif language == "en":
        model = spacy_EN
    else:
        model = spacy_EN
    return model, language

#get only "meaningful" text
def is_sentence(string, POS_model):
    subj, verb = 0, 0
    doc = POS_model(string)
    for token in doc:
        if token.dep_ in ["nsubj", "nsubj:pass", "csubj", "nsubjpass", "csubjpass"]:
            subj = 1
        elif token.pos_ in ["AUX", "VERB"]:
            verb = 1
    return bool(subj*verb)

def stop_substring(string):
    #url
    stop_expr = [(ele.start(), ele.end()) for ele in re.finditer(r'\S+', string) if bool(re.search('\.[a-z]+/', ele.group()))]
    #common
    stopwords = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",\
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
    stop_expr += ([(ele.start(), ele.end()) for ele in re.finditer(r'\S+', string) if ele.group().lower() in stopwords])
    
    return stop_expr
    
# get locs from a text using the NER model
def get_ner(string, NER_model, ner_list: list):
    stop_subs = stop_substring(string)
    ner = NER_model(string)
    locs = []
    for index, n in enumerate(ner):
        if n.ent_type_ in ner_list:
            if len(stop_subs) == 0 or not any([n.idx >= stop_sub[0] and n.idx+ len(n.orth_)<= stop_sub[1] for stop_sub in stop_subs]):
                locs.append({"ent":n.orth_, "offset_start": n.idx, "offset_end": n.idx + len(n.orth_)})
    
    locs = merge_multiwords(string, locs)

    return locs

def get_trans_ner(string, NER_model, ner_list: list):
    stop_subs = stop_substring(string)
    ner = NER_model(string)
    locs = []
    full_words = [(ele.group(), ele.start(), ele.end()) 
                  for ele in re.finditer(r'\b[A-Za-z_À-ÿöçğışü]+', string, re.IGNORECASE)]
    for index, n in enumerate(ner):
        if (n["entity"] in ner_list and bool(re.search('[a-zA-Z_À-ÿöçğışü]', n["word"], re.IGNORECASE))):
            if len(stop_subs) == 0 or not any([n["start"] >= stop_sub[0] and n["end"]<= stop_sub[1] for stop_sub in stop_subs]):
                try:
                    [full_word] = [(elem[0], elem[1], elem[2]) 
                                   for elem in full_words if n["start"]>= elem[1] and n["end"]<= elem[2]]

                    locs.append(full_word)
                except:
                    pass
                    
    locs = list(set(locs))

    def sort_fn(el):
        return el[1]
    locs.sort(key = sort_fn)
    locs = [{"ent": elem[0], "offset_start": elem[1], "offset_end": elem[2]} for elem in locs]

    locs = merge_multiwords2(string, locs)

    return locs

# merge multiwords locs unrecognised by NER model (e.g. ["New","York"] --> "New York")
def merge_multiwords(string, locs: list):
    for loc in reversed(range(1, len(locs))):
        if (locs[loc-1]["offset_end"] == locs[loc]["offset_start"] -1 and \
            string[locs[loc-1]["offset_end"]] in [' ', '-', '’', "'"]):
            
            sep = string[locs[loc-1]["offset_end"]]
            locs[loc-1]["ent"] = locs[loc-1]["ent"]+ sep + locs[loc]["ent"]
            locs[loc-1]["offset_end"] = locs[loc]["offset_end"]
            del locs[loc]
        elif(locs[loc-1]["offset_end"] == locs[loc]["offset_start"]):
            locs[loc-1]["ent"] = locs[loc-1]["ent"] + locs[loc]["ent"]
            locs[loc-1]["offset_end"] = locs[loc]["offset_end"]
            del locs[loc]
    
    return locs
# slightly different funtion to handle output from transformer model
def merge_multiwords2(string, locs: list):
    for loc in locs:
        loc["ent"] = re.sub("^▁", "", loc["ent"])
    for loc in reversed(range(1, len(locs))):
        
        if (locs[loc-1]["offset_end"] == locs[loc]["offset_start"] -1 and \
            string[locs[loc-1]["offset_end"]] in [' ', '-', '’']):
            
            sep = string[locs[loc-1]["offset_end"]]
            locs[loc-1]["ent"] = locs[loc-1]["ent"]+ sep + locs[loc]["ent"]
            locs[loc-1]["offset_end"] = locs[loc]["offset_end"]
            del locs[loc]
    
    return locs

# apply Spacy NER model to data    
def get_spacy_predictions(df, model = None):
    df_ = df.copy()
    if model == None:
        model = get_language(df_)[0]
    df_["sentence"] = df_["text"].apply(lambda x: is_sentence(x, model))
    df_["locs"] = df_["text"].apply(lambda x: get_ner(x, model, ner_list = ["GPE", "LOC"]))
    
    return df_

# apply Tranformers NER model to data
def get_transformer_predictions(df, model_path, model = None, tokenizer = None, mode = None):

    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from transformers import pipeline
    
    df_ = df.copy()
    if mode == 'reload':
        if model_path:
            trans_model = AutoModelForTokenClassification.from_pretrained(model_path)
            tokenizer_ = AutoTokenizer.from_pretrained(model_path)
            model_ = pipeline("ner", model=trans_model, tokenizer=tokenizer_)
        elif model:
            tokenizer_ = tokenizer
            model_ = pipeline("ner", model=model, tokenizer=tokenizer_)
    else:
        model_ = load_transformers_model(model_path)
    lang_model = get_language(df_)[0]
    df_["sentence"] = df_["text"].apply(lambda x: is_sentence(x, lang_model))
    df_["locs"] = df_["text"].apply(lambda x: get_trans_ner(x, model_, ner_list = ["B-LOC", "I-LOC"]))

    return df_

def get_uppercase_words(string):
    stop_words = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",\
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
    stop_words = stop_words.union({"age", "plan", "human", "national", "sector"})
    stop_words = stop_words.union(stopwords.words("english"))
    stop_words = stop_words.union(stopwords.words("french"))
    stop_words = stop_words.union(stopwords.words("spanish"))
    
    p = re.compile('(?<!^)(?<!\. )[A-Z][a-z]+')
    locs = []
    for m in p.finditer(string):
        if m.group().lower() not in stop_words:
            locs.append({'ent': m.group(), 'offset_start': m.start(), 'offset_end': m.end()})
    return locs

def get_candidates_locs(string, model, use_ner):
    
    candidates = get_uppercase_words(string)
    if use_ner:
        ner_list = ["MISC","ORG","PER", "CARDINAL", "DATE", "EVENT", "FAC", "LANGUAGE",\
            "LAW", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"]

        exclusions = get_ner(string, model, ner_list)

        for ent2 in exclusions:
            for ent in reversed(range(len(candidates))):
                if ent2["offset_start"] <= candidates[ent]["offset_start"] < ent2["offset_end"]:
                    del candidates[ent]
    return candidates

def get_custom_predictions(df, use_ner = True):
    df_ = df.copy()
    model = get_language(df_)[0]
    df_["locs"] = df_["text"].apply(lambda x: get_candidates_locs(x,model, use_ner))
    df_["locs"] = df_["locs"].apply(lambda x: match_with_gazetteer(x))
    return df_