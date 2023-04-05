import re
import ast
import json
import numpy as np
import pandas as pd
import unidecode
import logging

from collections import OrderedDict
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
from .geonameslocator import find_names


logging.getLogger().setLevel(logging.INFO)


def explode_locs(df, column = 'label'):
    '''
    input: DataFrame with rows = documents and field "label"(list) containing all the NER matches (entries) for that document
    output: exploded DataFrame with document/entry rows 
    '''
    df_c = df.copy()
    _df = df_c.loc[df_c.loc[:,column].astype(bool),:].explode(column)
    res = pd.concat([_df.loc[:, _df.columns != column], _df.loc[:,column].apply(pd.Series)], axis=1)
    res = res[["lead_id","text", "start", "end", "labels"]]
    res["labels"] = res["labels"].apply(lambda x: ' '.join(map(str, x)))
    return res

# post process functions
def split_lists_nounModifiers(string, plurals):
        #example: New York and Boston cities
    plurals_list = list(plurals.keys())
    singulars_list = list(plurals.values())
    if type(string) == str:
        string = [string]
    output = []
    for entry in string:
        try:
            ls = re.split(', and |,| and ', entry.lower())
            ls = [elem.strip() for elem in ls]
            if len(ls)>1:
                noun = [ele for ele in plurals_list if(' '+ele in ls[-1].lower())][0]
                ls = [elem + ' '+plurals[noun] for elem in ls]
                ls[-1] = re.split(' '+noun, ls[-1], flags=re.IGNORECASE, maxsplit = 1)[0] + ' ' + plurals[noun]
            output.extend(ls)
        except:
            try:
                ls = re.split(', and |,| and ', entry.lower())
                ls = [elem.strip() for elem in ls]
                if len(ls) >1:
                    noun = [ele for ele in singulars_list if(' '+ele in ls[-1].lower())][0]
                    ls = [(elem + ' '+noun).strip() for elem in ls]
                    ls[-1] = (re.split(' '+noun, ls[-1], flags=re.IGNORECASE, maxsplit = 1)[0] + ' ' + noun).strip()
                output.extend(ls)
            except:
                output.append(entry.strip())
    return output

def split_lists_PossesivePronouns(string, plurals):
    #example: Cities of New York and Boston
    if type(string) == str:
        string = [string]
    output = []
    for entry in string:
        try:
            if '(' not in entry:
                noun = re.split(' of ', entry.lower(), 1)[0].strip()
                possesive = re.split(', and |,| and ', re.split(' of ', entry.lower(), 1)[1].strip())
            output.extend([elem + ' ' + plurals[noun.lower()] for elem in possesive])
        except:
            try:
                if '(' not in entry:
                    noun = re.split(' of ', entry.lower(), 1)[0].strip()
                    possesive = re.split(', and |,| and ', re.split(' of ', entry.lower(), 1)[1].strip())
                output.extend([elem + ' ' + noun.lower() for elem in possesive])
            except:
                output.append(entry)
    return output

def split_lists_Cardinals(string, cardinals):
    if type(string) == str:
        string = [string]
    output = []
    for entry in string:
        try:
            ls = re.split(', and |,| and ', entry.lower())
            if all([elem.strip().lower() in cardinals for elem in ls[0:-1]]) and len([elem.strip().lower() in cardinals for elem in ls[0:-1]])>0:
                right = [ele.strip().lower() for ele in cardinals if(ele in ls[-1].lower())][0]
                if right in cardinals:
                    noun = re.split(right, ls[-1], flags=re.IGNORECASE, maxsplit = 1)[1].strip()
                    output = [elem.strip() + ' ' + noun for elem in ls[0:-1]]
                    output.append(right + ' ' + noun)
            elif all([elem.strip().lower() in cardinals for elem in ls[1:]]) and len([elem.strip().lower() in cardinals for elem in ls[1:]])>0:
                left = [ele.strip().lower() for ele in cardinals if(ele in ls[0].lower())][0]
                if left in cardinals:
                    noun = re.split(left, ls[0], flags=re.IGNORECASE, maxsplit = 1)[0].strip()
                    output.append(left + ' ' + noun)
                    output.extend([elem.strip() + ' ' + noun for elem in ls[1:]])
            else: 
                output.append(entry)
        except: 
            output.append(entry)
                
    return output

def GetUniqueEntities(data):
    entities = pd.DataFrame(data)
    # filter only locs (exclude Associative "AST")
    entities.rename(columns = {"ent":"original", "offset_start":"start", "offset_end":"end"}, inplace = True)
    
    #list of attributes, cardinal directions
    plurals = OrderedDict({
        "provinces": "province",
        "towns": "town",
        "villages": "village",
        "hospitals": "hospital",
        "sea ports": "sea port",
        "ports": "port",
        "districts": "district",
        "cities": "city",
        "islands": "island",
        "states": "state",
        "municipalities": "municipality",
        "governorates": "governorate",
        "axes": "axis",
        "regions": "region",
        "hubs": "hub",
        "continents": "continent",
        "idp camps": "idp camp",
        "refugee camps": "refugee camp",
        "camps": "camp",
        "areas": "area",
        "sub-districts": "sub-district",
        "subdistricts": "subdistrict",
        "sub districts": "sub district",
        "residential complexes": "residential complex",
        "basins": "basin",
        "administrative centres": "administrative centre",
        "administrative centers": "administrative center",
        "detention centres": "detention centre",
        "detention centers": "detention center",
        "centres": "centre",
        "centers": "center",
        "subdivisions": "subdivision",
        "oil refineries": "oil refinery",
        "territories": "territory",
        "lgas": "lga",
        "divisions": "division",
        "countries": "country",
        "checkpoints": "checkpoint",
        "rivers": "river",
        "markets": "market",
        "departments": "department",
        "airports": "airport",
        "local government areas": "local government area",
        "bus stations": "bus station",
        "military camps": "military camp",
        "subdivisions": "subdivision",
        "highways": "highway",
        "extensions": "extension",
        "capitals": "capital",
        "border points": "border point",
        "crossing points": "crossing point",
        "borders": "border",
        "roads": "road",
        "channels": "channel",
        "mountains": "mountain",
        "prefectures": "prefecture",
        "volcanoes": "volcano",
        "volcanos": "volcano",
        "highlands": "highland",
        "lakes": "lake"
    })
    cardinals = ["northeast", "north-east", "northwest", "north-west", "northouest", "north-ouest", \
                 "north east", "north west", "south east", "south west", "southeast","south-east","southwest","south-west","southouest","south-ouest", \
                 "north","east", "west", "ouest","south", "central"]
    cardinals_adj = ["northern", "northestern", "north-estern", "northwestern", "north-western", "northouestern", "north-ouestern", \
                 "eastern", "western", "ouestern","southern","southeastern","south-eastern","southwestern","south-western","southouestern","south-ouestern",
                    "centre", "central"]
    
    entities["post_entities"] = entities["original"].apply(lambda x: split_lists_nounModifiers(x, plurals))
    entities["post_entities"] = entities["post_entities"].apply(lambda x: split_lists_PossesivePronouns(x, plurals))
    entities["post_entities"] = entities["post_entities"].apply(lambda x: split_lists_Cardinals(x, cardinals))
    
    #explode lists of entities
    entities = entities.explode("post_entities")
    #uniques
    unique_entities = entities.groupby(["original","post_entities"]).post_entities.agg('count').to_frame('freq').reset_index()
    
    return unique_entities

def loadGeonames(locationdata_path,locdictionary_path,reload):
    
    #locationdata_path = "/data/fast/ebelliardo/geonames/locationdata_large.tsv" if locationdata_path == None else locationdata_path
    #locdictionary_path = "/data/fast/ebelliardo/geonames/locdictionary_unicode.json" if locdictionary_path == None else locdictionary_path
    
    global locationdata
    global locdictionary
    
    if reload:
        logging.info("load locationdata...")
        locationdata = pd.read_csv(locationdata_path, sep='\t', low_memory=False, index_col="geonameid")
        locationdata.loc[locationdata['alternatenames'].isnull(),'alternatenames'] = '[]'
        locationdata["alternatenames"] = locationdata["alternatenames"].apply(lambda x: ast.literal_eval(x))
        
        logging.info("load locdictionary...")
        locdictionary = json.load(open(locdictionary_path))
    else:
        logging.info("load locationdata...")
        try:
            locationdata
        except NameError:
            locationdata = pd.read_csv(locationdata_path, sep='\t', low_memory=False, index_col="geonameid")
            locationdata.loc[locationdata['alternatenames'].isnull(),'alternatenames'] = '[]'
            locationdata["alternatenames"] = locationdata["alternatenames"].apply(lambda x: ast.literal_eval(x))
        
        logging.info("load locdictionary...")
        try:
            locdictionary
        except NameError:
            locdictionary = json.load(open(locdictionary_path))
        return locationdata, locdictionary
    
# Create a query
def search_index(querystring:str, parser, searcher):
    query = parser.parse(querystring)
    # Run the search
    results = searcher.search(query)
    #return set([int(r['geonameid']) for r in results if r["place"].lower()!= querystring])
    return set([int(r['geonameid']) for r in results])


def match_search(df, column,locationdata, match_string, parser, searcher):
    match_search = [list(search_index(x, parser, searcher)) for x in df[column].tolist()]
    match_flat = list(set([e for x in match_search for e in x]))

    match_df = locationdata[locationdata.index.isin(match_flat)]
    filter_list = match_df[match_df.featurecode.str.contains(match_string, na = False)].index.tolist()

    return [[e for e in x if e in filter_list] for x in match_search]

def UniqueEntities_fromLS(data, locationdata_path = None, locdictionary_path = None, use_search_engine = True, 
                          indexdir = None, filter_search_engine = '', reload = False):
    
    #get unique entities
    unique_entities = GetUniqueEntities(data)
    
    #load geonames
    locationdata, locdictionary = loadGeonames(locationdata_path, locdictionary_path, reload)
    
    unique_entities["post_entities"] = unique_entities["post_entities"].apply(lambda x: unidecode.unidecode(x))
    unique_entities["post_entities"] = unique_entities["post_entities"].apply(lambda x: re.sub("-|'", "", x))
    
    logging.info("apply geonames...")
    unique_entities["match"] = unique_entities["post_entities"].apply(lambda x: find_names(x, locdictionary, locationdata))

    # exclude cardinality
    cardinals = ["northeast", "north-east", "northwest", "north-west", "northouest", "north-ouest", \
                 "north east", "north west", "south east", "south west", "southeast","south-east","southwest","south-west","southouest","south-ouest", \
                 "north","east", "west", "ouest","south", "central"]
    unique_entities["match"] = unique_entities['match'].map(set) - set(cardinals) -set(['of'])
    
    # apply dict
    unique_entities = unique_entities.explode("match")
    unique_entities["geonameid"] = unique_entities["match"].apply(lambda x: locdictionary[x] if pd.notnull(x) else [])
    
    if use_search_engine:
        logging.info("add search engine results...")
        indexdir = "geonames/indexdir" if indexdir == None else indexdir
        ix = open_dir(indexdir)
        # Create a searcher of that index
        searcher = ix.searcher()
        # Prepare a parser
        parser = QueryParser("place", ix.schema)
        
        unique_entities["match_search"] = pd.Series(match_search(unique_entities, "post_entities", locationdata, 
                                                                 filter_search_engine, parser,searcher)).values
        unique_entities["geonameid_all"] = unique_entities.apply(lambda x: list((set(x.geonameid).union(set(x.match_search)))), axis=1)
        unique_entities['from_search'] = pd.Series((unique_entities.match_search.map(set) - unique_entities.geonameid.map(set)).map(list))
        unique_entities = unique_entities.drop(columns = {"geonameid", "match_search"}).rename(columns = {"geonameid_all":"geonameid"})
    else:
        unique_entities['from_search'] = unique_entities.np.empty((len(df), 0)).tolist() #error here
        
    return unique_entities

def BuildTarget(unique_entities, gt_path, 
                locationdata_path = None, 
                locdictionary_path = None,
                reload = False):
    locationdata, locdictionary = loadGeonames(locationdata_path, locdictionary_path, reload)
    
    if gt_path != None:
        with open(gt_path) as f:
            gt_json = json.load(f)

        gt = pd.DataFrame.from_records(
        [
            (level1, level2,level3)
            for level1, level2_dict in gt_json.items()
            for level2, level3 in level2_dict.items()
        ],
        columns=['match', 'geonameid'], 
                 )
        gt.lead_id = gt.lead_id.astype("float").astype("Int64")
        gt['is_toponym'] = gt.apply(lambda x: 0 if x.geonameid == -1 else 1, axis = 1)
    
    target = unique_entities[["post_entities","match", "geonameid", "from_search"]].explode("geonameid")
    target['from_search'] = target.apply(lambda x: (x.geonameid in x.from_search)*1, axis = 1)
    target = target[target.geonameid.notnull()]
    if gt_path != None:
        idx = list(target.merge(gt, on = ["match"], how = "left", indicator = True)._merge == "both")
        target = target[idx]
        target = target.merge(gt[['match','is_toponym']].drop_duplicates(), on = 'match', how = 'left')
        target = target.merge(gt[['match', 'geonameid']], on = ["match","geonameid"], how = "left", indicator = True)
        target["target"] = target["_merge"].map({"left_only":0, "both":1})
        target.drop(columns = ["_merge"], inplace = True)
    else:
        target["target"] = 0
        target["is_toponym"] = None
    target = target.merge(locationdata, right_index=True, left_on='geonameid')
    target.geonameid = target.geonameid.astype("int")
    target.latitude = np.radians(target.latitude)
    target.longitude = np.radians(target.longitude)
    
    return target