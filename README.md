
Perform geo entities detection and disambiguation in respect of GeoNames database

# Install

```pip install git+https://github.com/the-deep-nlp/geolocation-generator.git```

It's possibile to install an extension compatible with the transformers library: `geolocation_generator[transformers]`
Default version runs with SpaCy library models.
# Usage (API)

For a faster version of the library, please use this method based on the Geonames search system (http://www.geonames.org/export/geonames-search.html). A little disambiguation ranking is builded on top of it (based on result countries distribution, geonames output order and entities types). In order to use it, please create a free account on http://www.geonames.org/login 
```
from geolocation_generator import GeolocationGenerator

geolocation = GeolocationGenerator(spacy_path=MODEL_NAME_OR_PATH)

results = geolocation.get_geolocation_api(
                    raw_data = YOUR_DATA,
                    geonames_username=YOUR_GEONAMES_USERNAME,
                    geonames_token=YOUR_GEONAMES_TOKEN or None,
                    premium_service=True or False
                )

```
# Usage

```
from geolocation_generator import GeolocationGenerator

geolocation = GeolocationGenerator(spacy_path=MODEL_NAME_OR_PATH) # trans_path is set to None, install the extension to use it

results = geolocation.get_geolocation(
                    raw_data = YOUR_DATA,
                    locationdata_path = GEONAMES_LOCATIONDATA,
                    locdictionary_path = GEONAMES_LOCATION_DICTIONARY,
                    indexdir = CUSTOM_INDEX
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
