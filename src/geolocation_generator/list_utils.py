import ast
from nltk.corpus import stopwords

def apply_nList(item, fun):
    '''applies generic function to innermost nested list'''
    if isinstance(item, list):
        return [apply_nList(x, fun) for x in item]
    else:
        return fun(item)
    
def apply_nListS(a, b, fun):
    '''apply fun to innermost level of nested list from 2 lists of same shape'''
    if len(a)>0 and isinstance(a[0], list):
        return [apply_nListS(item_a, item_b, fun) for item_a, item_b in zip(a, b)]
    else:
        return fun(a,b)
    
def concat(a:list,b:list): return a + b

def flatten_nList(a):
    '''flatten arbitrary nested lists '''
    for i in a:
        if isinstance(i, (list,tuple)):
            for j in flatten_nList(i):
                yield j
        else:
            yield i
            
def addText_nListS(a, b):
    
    if isinstance(b, list):
        return [addText_nListS(item_a, item_b) for item_a, item_b in zip(a, b)]
    else:
        return {'text':b, 'entities':a}

def matchGeoids_nListS(a, loc_dict):
    if isinstance(a, list):
        return [matchGeoids_nListS(x, loc_dict) for x in a]
    else:
        if loc_dict[a["ent"]]:
            a['geoids'] = loc_dict[a['ent']]
        return a
    
def cleanFinalOutput(x):
    
    def cast_geoids(x1):
        x1["geoids"] = ast.literal_eval(x1["geoids"])
        return x1

    stop_words = stopwords.words("english") + stopwords.words("french") + stopwords.words("spanish")
    final = []
    for el in x:
        if el["entities"]:
            ents = [cast_geoids(c) for c in el["entities"] if c.get("geoids") and c["ent"].lower() not in stop_words]
            el["entities"] = ents
        final.append(el)
    
    return final