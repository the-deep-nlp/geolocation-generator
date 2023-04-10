import spacy
import pandas as pd
import nltk
from .list_utils import apply_nList
from .NER_models import get_ner, get_trans_ner
from .LS_utils import merge_overlapping_ents
from .list_utils import concat, flatten_nList, addText_nListS, matchGeoids_nListS, apply_nListS, cleanFinalOutput
from .geomatch import UniqueEntities_fromLS, BuildTarget
from .disambiguation_algorithms import RuleBasedDisambiguation

nltk.data.path.append('/nltk_data')

class GeolocationGenerator:

    def __init__(self, spacy_path: str = None, transformers_path: str = None):
        
        self.spacy_model_path = spacy_path
        self.trans_model_path = transformers_path
        
        if spacy_path:

            self.model_spacy = spacy.load(self.spacy_model_path)
        
        if transformers_path:
            
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

            roberta_model = AutoModelForTokenClassification.from_pretrained(self.trans_model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.trans_model_path)
            self.model_trans = pipeline("ner", model=roberta_model, tokenizer=tokenizer)

    def shape_data(self, data):

        locs = apply_nList(data, lambda x: get_ner(x, self.model_spacy, ["GPE", "LOC"]))
        if self.trans_model_path:
            locs_r = apply_nList(data, lambda x: get_trans_ner(x, self.model_trans, ["B-LOC", "I-LOC"]))
            locs = apply_nList(locs, locs_r, concat)
        
        merged = apply_nListS(locs, data, merge_overlapping_ents)

        return merged, pd.DataFrame(flatten_nList(merged))

    @staticmethod
    def get_entities(data: pd.DataFrame,
                     locationdata_path:  str,
                     locdictionary_path: str,
                     indexdir: str,
                     use_search_engine: bool = False,
                     filter_search_engine: str = '^AD|^PPL|^MT|^SEA|^LK$|^ISL$|^AIR',
                     reload: bool = False):
        
        entities = UniqueEntities_fromLS(data,
                                        locationdata_path=locationdata_path,
                                        locdictionary_path=locdictionary_path,
                                        use_search_engine=use_search_engine,
                                        indexdir=indexdir,
                                        filter_search_engine=filter_search_engine,
                                        reload=reload)
        
        target = BuildTarget(entities,
                             gt_path=None,
                             locationdata_path=locationdata_path,
                             locdictionary_path=locdictionary_path,
                             reload=reload)
        
        target["lead_id"] = 0
        disambiguated = RuleBasedDisambiguation(target)
        disambiguated = disambiguated[disambiguated.final_pred == 1]
        disambiguated[['post_entities','match', 'geonameid']]

        final = entities[['original', 'match','post_entities']].merge(disambiguated[['post_entities', 
                                                                                     'match', 
                                                                                     'geonameid', 
                                                                                     'latitude', 
                                                                                     'longitude', 
                                                                                     'featurecode', 
                                                                                     'countrycode']], on = ["post_entities", "match"], how = "left")
        final['geonameid'] = final['geonameid'].astype("Int64")
        loc_dict = final.groupby(['original']).apply(lambda x: x[['match', 
                                                                  'geonameid', 
                                                                  'latitude', 
                                                                  'longitude', 
                                                                  'featurecode', 
                                                                  'countrycode']].to_json(orient='records') if x["geonameid"].notnull().all()  else []).to_dict()


        return loc_dict
    
    def get_geolocation(self, 
                       raw_data,
                       locationdata_path:  str,
                       locdictionary_path: str,
                       indexdir: str,
                       use_search_engine: bool = False,
                       filter_search_engine: str = '^AD|^PPL|^MT|^SEA|^LK$|^ISL$|^AIR',
                       reload: bool = False
                       ):
        
        merged, locs_df = self.shape_data(data=raw_data)

        loc_dict = GeolocationGenerator.get_entities(data=locs_df,
                                                     locationdata_path=locationdata_path,
                                                     locdictionary_path=locdictionary_path,
                                                     use_search_engine=use_search_engine,
                                                     indexdir=indexdir,
                                                     filter_search_engine=filter_search_engine,
                                                     reload=reload)
        
        final = cleanFinalOutput(addText_nListS(matchGeoids_nListS(merged, loc_dict), raw_data))
        return final