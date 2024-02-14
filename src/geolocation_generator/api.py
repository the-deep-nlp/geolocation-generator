
import re 
import requests
import logging
from typing import Optional
from collections import Counter
from requests.exceptions import Timeout
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.getLogger().setLevel(logging.INFO)


cardinal_expression_pattern = re.compile(r"(?:northeast|northwest|southeast|southwest|north|south|east|west)(?:|ern)(?:\-|\s|)(?:north|west|east|south|)(?:ern|)", re.IGNORECASE)
GEONAMES_REQUESTS = "http://api.geonames.org/searchJSON?q={}&maxRows=10&featureClass=A&featureClass=P&featureClass=L&featureClass=S&featureClass=S&username={}"
GEONAMES_REQUESTS_PREMIUM = "http://ws.geonames.net/searchJSON?q={}&maxRows=10&featureClass=A&featureClass=P&featureClass=L&featureClass=S&featureClass=S&username={}&token={}"


def clean_geo_tag(x):
    
    if x=="nan":
        return []
    def cardinal(x1):
        if x1.lower()=="nws":
            x1 = "northwest syria"
        find = cardinal_expression_pattern.findall(x1)
        for c in find:
            x1 = x1.replace(c, "").strip()
        if x1:
            return (x1, [c.strip() for c in find])
    splits = [cardinal(c.strip()) for c in re.split(r';|,|&|and|part of| in |greater |\/', x.get("ent")) if c]
    x.update({"clean": [a for a in splits if a and a[0] not in [None, "-"]]})
    return splits

def req_handler(data: dict, timeout: int=10):
    try:
        if data["premium_service"] and data["token"]:
            response = requests.get(
                GEONAMES_REQUESTS_PREMIUM.format(data["point"], data["username"], data["token"]),
                timeout=timeout
            )
        else:
            response = requests.get(
                GEONAMES_REQUESTS.format(data["point"], data["username"]),
                timeout=timeout
            )
    except Timeout:
        logging.error("Timeout to Geolocation API has occurred.")
        response = None
    except Exception as exc:
        logging.error("Error has occurred while sending request. %s", str(exc))
        response = None
    return {
        "response": response.json() if response and response.status_code==200 else None,
        "loc": data["loc"],
        "status_code": response.status_code if response else -1
    }

def build_dict(x: list, username: str, token: Optional[str], premium_service: bool):
    dict_final = {}

    data = list(set([(loc, a[0].lower()) for loc, point in x for a in point if a]))
    futs = []
    for loc, point in data:
        if point in dict_final.keys() or len(point)<=3:
            continue
        with ThreadPoolExecutor(max_workers=5) as executor:
            futs.append(executor.submit(
                req_handler,
                data={
                    "point": point,
                    "loc": loc,
                    "username": username,
                    "token": token,
                    "premium_service": premium_service
                })
            )
        results = [fut.result() for fut in as_completed(futs)]
        
        for result in results:
            if result["response"]:
                if result["response"]["totalResultsCount"] > 0:
                    dict_final.update({
                        result["loc"]: result["response"]["geonames"]
                    })
            else:
                logging.error(f"Some error occurs requesting GeoNames API. Status Code: {result['status_code']}")
    
    dict_final = reshape_final(dict_final)
    return dict_final


def normalize(data, min_, max_):
    if min_ == max_:
        return data
    return (data - min_) / (max_ - min_)


def reshape(counter):
    res = {}
    values = list(counter.values())
    if not len(values):
        return res
    min_, max_ = min(values), max(values)
    for k, v in counter.items():
        res.update({k: normalize(v, min_, max_)})
    return res


def get_scores(group, counter, min_threshold: float = 0.5, offset: float = 0.1):

    for i,v in enumerate(group[::-1]):
        country_score = counter.get(v.get("countryName"))
        order = normalize(i+1, 1, len(group)+offset)
        if v.get("fclName") in [
            "country, state, region,...",
            "city, village,...",
            "parks,area, ..."
        ]:
            types = 1
        else:
            types = 0
        v["score"] = country_score + order*min_threshold + types*min_threshold
    first = sorted(group, key=lambda x: x["score"], reverse=True)[0]

    return first if first.get("score")>min_threshold else []


def reshape_final(geodict):
    
    final = {}
    counter = Counter([c.get("countryName")
                       for x in geodict.values() for c in x])
    counter = reshape(counter)
    for k, v in geodict.items():
        selected = get_scores(v, counter)
        if selected:
            final[k] = selected
        
    return final


def format_output(geodictionary, raw_data, entities):

    final_results = [
        {
            "text": loc,
            "entities": [{
                "ent": x.get("ent"),
                "offset_start": x.get("offset_start"),
                "offset_end": x.get("offset_end"),
                "geoids": [
                    {
                        "match": geodictionary.get(x.get("ent"), {}).get("name"),
                        "geonameid": geodictionary.get(x.get("ent"), {}).get("geonameId"),
                        "latitude": float(geodictionary.get(x.get("ent"), {}).get("lat")),
                        "longitude": float(geodictionary.get(x.get("ent"), {}).get("lng")),
                        "featurecode": geodictionary.get(x.get("ent"), {}).get("fcode"),
                        "contrycode": geodictionary.get(x.get("ent"), {}).get("countryCode")
                    }
                ]
            }
                
        for x in point if geodictionary.get(x.get("ent"))]}
        for loc, point in zip(raw_data, entities)]
    
    return final_results