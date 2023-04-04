
Perform geo entities detection and disambiguation in respect of GeoNames database

# Install

```pip install ```

It's possibile to install an extension compatible with the transformers library: `geolocation_generator[transformers]`
Teh default version runs with SpaCy library models.

# Usages

```
from geolocation_generator import GeolocationGenerator

geolocation = GeolocationGenerator(spacy_path=MODEL_NAME_OR_PATH) # trans_path is set to None, install the extension to use it

results = geolocation.get_geolocation(
                    raw_data = YOUR_DATA,
                    locationdata_path = GEONAMES_LOCATIONDATA,
                    locdictionary_path = GEONAMES_LOCATION_DICTIONARY,
                    indexdir = CUSTOM_INDEX"
                    use_search_engine = True
                )
```
`YOUR_DATA` is a list of entries (list of str). 

Output example:

```
 [
    {
        'text': ENTRY_TEXT,
        'entities': [
            {
                'ent': 'Colombia',
                'offset_start': 29,
                'offset_end': 37,
                'geoids': [{
                    'match': 'colombia',
                    'geonameid': 3686110,
                    'latitude': 0.0698131701,
                    'longitude': -1.2784536771,
                    'featurecode': 'PCLI',
                    'countrycode': 'CO'
                    }
                ]
            },
            {
                'ent: ....
            },
            ...
        ]
    },
    ...
 ]

```
Final output is a list of elements like the one in the example. The output list order will follow the one of the entries list input.