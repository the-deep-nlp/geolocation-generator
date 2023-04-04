import numpy as np
import pandas as pd

def get_coord(idx, locationdata):  
    '''get coordinates givel list of ids'''
    try:
        res =  np.radians(np.array(locationdata.loc[idx][["latitude", "longitude"]], dtype = 'float64'))
    except:
        res = None
    return res

def great_circle(x,y):
    '''
    given 2 pairs of coordinates (lat_x,lon_x), (lat_y, lon_y) computes the distance in Km
    '''
    try:
        cos = np.minimum(np.sin(x[0]) * np.sin(y[0]) + np.cos(x[0]) * np.cos(y[0]) * np.cos(x[1] - y[1]), 1.)
        res =  6371 * (np.arccos(cos))
    except:
        res = None
    return res

def geoDistance(df, locationdata):
    '''
    compute distance between predicted and benchmark
    '''
    df["coord_true"] = df["geonameid_true"].apply(lambda x:get_coord(x, locationdata))
    df["coord_pred"] = df["geonameid_pred"].apply(lambda x:get_coord(x, locationdata))
    
    df["dist"] = df.apply(lambda x: great_circle(x.coord_true, x.coord_pred), axis = 1)
    df = df.drop(["coord_true", "coord_pred"], axis = 1)
    return df