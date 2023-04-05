import logging
import numpy as np
import pandasql as ps
from .geo_distance import great_circle
from sklearn.linear_model import LogisticRegression


logging.getLogger().setLevel(logging.INFO)


def RuleBasedDisambiguation(target, use_search_engine = False):
    
    relevant_columns = ['lead_id', 'post_entities','match', 'geonameid', 'target',
                        'latitude', 'longitude','featurecode', 'countrycode',
                        'population', 'from_search', 'is_toponym']

    data = target[relevant_columns].copy()
    if not use_search_engine:
        data = data[~data.from_search.astype(bool)]

    #compute document/country distribution:
    sqlcode = '''
    select lead_id, countrycode, sum(prop) distr
    from(
        select lead_id, match, countrycode, countn, 1.0*countn/ sum(countn) over (partition by lead_id, match) prop
        from(
            select lead_id, match, countrycode, count(1) countn
            from data
            group by lead_id, match, countrycode
            ) a
        )b
    group by lead_id, countrycode
    order by lead_id, distr desc
    '''

    doc_country_distr = ps.sqldf(sqlcode, locals())

    sqlcode = '''
    select a.*, b.distr country_distr
    from data a
    join doc_country_distr b
    on a.lead_id = b.lead_id and a.countrycode = b.countrycode
    order by lead_id'''

    data = ps.sqldf(sqlcode, locals())

    #is country/capital city?
    data['is_country'] = (data.featurecode.isin(["PCLI", "PCLS"]) & (data.post_entities == data.match))*1 
    data['is_capital'] = (data.featurecode.isin(["PPLC"]) & (data.post_entities == data.match))*1 

    #add admin tag: 0:high --> 6:low
    data['adm_level'] = data["featurecode"].str.findall('\d+').str.join("").replace("", 6)
    data['adm_level'] = data["adm_level"]*(1-data["is_capital"])*(1-data["is_country"])
    data.loc[data.featurecode == "PPLA", "adm_level"] = 1
    data["adm_level"] = data["adm_level"].fillna(6)

    data["is_city"] = data["featurecode"].str.contains("PPL")*1


    #rank by
    # - is_capital/is_country
    # - country_distr
    # - population
    # - admin_tag
    #

    sqlcode = '''
    select *, 
        row_number() over (partition by lead_id, match 
                            order by is_capital desc, 
                            is_country desc, 
                            country_distr desc, 
                            adm_level,
                            is_city desc,
                            population desc) as rankn
    from data
    '''
    data = ps.sqldf(sqlcode, locals())
    
    #thresholds

    sqlcode = '''
    select distinct * from (
    select lead_id, countrycode, rank() over( partition by lead_id order by country_distr desc) rk
    from data )
    where rk =1
    '''
    first_country = ps.sqldf(sqlcode, locals())
    
    sqlcode = '''
    select a.* from data a
    left join first_country b on a.lead_id = b.lead_id
    where (a.adm_level < 5) or (a.countrycode = b.countrycode)'''

    data = ps.sqldf(sqlcode, locals())

    #pick first of the ranking
    data['final_pred'] = data['rankn'].apply(lambda x: 1. if x == 1 else 0)
    
    return data

def TrainCustomDisambiguation(target, iters, use_search_engine):
    relevant_columns = ['lead_id', 'post_entities','match', 'geonameid', 'target',
                        'latitude', 'longitude','featurecode', 'countrycode',
                        'population', 'from_search', 'test_set']

    data = target[relevant_columns].copy()
    if not use_search_engine:
        data = data[~data.from_search.astype(bool)]

    #is country? is populated? is capital? is city?
    data['is_country'] = data.featurecode.isin(["PCLI", "PCLS"])*1
    data['is_pop'] = (data.population>1)*1
    data['is_capital'] = data.featurecode.isin(["PPLC"])*1

    data["is_city"] = data["featurecode"].str.contains("PPL")*1
    data["is_city"] = data["is_city"].fillna(0)

    #add admin tag: 0:high --> 6:low
    data['adm_level'] = data["featurecode"].str.findall('\d+').str.join("").replace("", 6)
    data['adm_level'] = data["adm_level"]*(1-data["is_capital"])*(1-data["is_country"])
    data.loc[data.featurecode == "PPLA", "adm_level"] = 1
    data["adm_level"] = data["adm_level"].fillna(6)
    data.adm_level = data.adm_level.astype(int)

    #run first regression on features. prediction_proba will be used as weights
    model = LogisticRegression()
    model.class_weight = "balanced"

    X = data[["is_pop","adm_level"]]
    X_train = data.loc[data.test_set == 0, ["is_pop","adm_level"]]
    Y_train = data.loc[data.test_set == 0,"target"]
    model.fit(X_train, Y_train)
    logging.info("initial coeff:", model.intercept_, model.coef_)

    data['predict'] = model.predict_proba(X)[:,1]

    # compute pairwise distance between candidates
    logging.info("compute pairwise distance ---------")
    sqlcode = '''
    select a.lead_id, a.match name1, a.geonameid geonameid1, a.latitude latitude1, a.longitude longitude1,
        b.match name2, b.geonameid geonameid2, b.latitude latitude2, b.longitude longitude2 
    from data as a
    join data as b
    on a.lead_id = b.lead_id and a.match <> b.match'''

    pairs = ps.sqldf(sqlcode, locals())
    pairs['coordinate1'] = list(zip(pairs.latitude1, pairs.longitude1))
    pairs['coordinate2'] = list(zip(pairs.latitude2, pairs.longitude2))

    pairs["distance"] = pairs.apply(lambda x: great_circle(x.coordinate1, x.coordinate2), axis = 1)
    pairs.drop(columns = ['coordinate1', 'coordinate2'], inplace = True)

    for i in range(iters):
        #is highest ranked candidate so far?
        sqlcode = '''
                select *,
                    case when best == 1 then 1
                         when best <> 1 then 0
                    end as isbest
                from(
                select *, rank() over(partition by lead_id, match order by predict desc) best from data) a
                '''    
        data = ps.sqldf(sqlcode, locals())

        #country distribution (weighted)
        sqlcode = '''
        select * from (
        select lead_id, match, countrycode, predict,
        rank() over (partition by lead_id, match, countrycode order by target desc) rankn
        from data)a
        where rankn = 1
        '''
        country_distr = ps.sqldf(sqlcode, locals())

        sqlcode = '''
        select lead_id, countrycode, sum(sumw) sumw
        from(
        select lead_id, match, countrycode, predict / sum(predict) over(partition by lead_id, match) sumw
        from country_distr)a
        group by lead_id, countrycode
        order by lead_id, sum(sumw) desc'''
        country_distr = ps.sqldf(sqlcode, locals())
        country_distr.sumw = country_distr.groupby("lead_id")['sumw'].transform(lambda x: (x - x.mean()) / x.std())

        sqlcode = '''
        select a.*, b.sumw
        from data a
        join country_distr b on a.lead_id = b.lead_id and a.countrycode = b.countrycode'''

        data = ps.sqldf(sqlcode, locals())

        #average distance from other candidates (weighted)
        sqlcode = '''
        select m2.lead_id, m2.name1, m2.geonameid1, avg(m2.wdistance) wdistance
        from
        (
        select m.lead_id, m.name1, m.name2, m.geonameid1, sum(m.distance*m.distance*m.weight)/sum(m.weight) wdistance
        from (
            select a.lead_id, a.name1, a.name2, a.geonameid1, a.geonameid2, a.distance, b.predict weight
            from pairs a
            join data b on a.lead_id = b.lead_id and a.geonameid2 = b.geonameid
            ) m
        group by m.lead_id, m.name1, m.name2, m.geonameid1
        ) m2
        group by
        m2.lead_id, m2.name1, m2.geonameid1
        '''

        wdistance = ps.sqldf(sqlcode, locals())

        sqlcode = '''
        select a.lead_id, a.match, a.geonameid, latitude, longitude,
           population, featurecode, target, is_country, is_pop,
           is_capital, adm_level, is_city, predict, sumw, countrycode, isbest, test_set,
           b.wdistance 
           from data a
        join wdistance b on a.lead_id = b.lead_id and a.geonameid = b.geonameid1 and a.match = b.name1'''

        data = ps.sqldf(sqlcode, locals())
        data["wdistance"] = data.groupby('lead_id')["wdistance"].transform(lambda x: (x - x.mean()) / x.std())

        #is closest candidate so far?
        sqlcode = '''
                select *,
                    case when closest == 1 then 1
                         when closest <> 1 then 0
                    end as isclosest
                from(
                select *, row_number() over(partition by lead_id, match order by wdistance) closest from data) a
                '''    
        data = ps.sqldf(sqlcode, locals())

        # compute regression with new features
        X = data[["is_pop","adm_level", "wdistance", "sumw", "isbest", "isclosest"]]
        X_train = data.loc[data.test_set == 0, ["is_pop","adm_level", "wdistance", "sumw", "isbest", "isclosest"]]
        Y_train = data.loc[data.test_set == 0,"target"]

        model.fit(X_train, Y_train)
        logging.info("coeff:", model.intercept_, model.coef_)
    return model.intercept_, model.coef_

def sig(x):
    return 1/(1 + np.exp(-x))

def PredictCustomDisambiguation(target, coeffs, iters, use_search_engine = False):
    relevant_columns = ['lead_id', 'post_entities','match', 'geonameid', 'target',
                    'latitude', 'longitude','featurecode', 'countrycode',
                    'population', 'from_search', 'test_set']

    data = target[relevant_columns].copy()
    if not use_search_engine:
        data = data[~data.from_search.astype(bool)]

    #is country? is populated? is capital? is city?
    data['is_country'] = data.featurecode.isin(["PCLI", "PCLS"])*1
    data['is_pop'] = (data.population>1)*1
    data['is_capital'] = data.featurecode.isin(["PPLC"])*1

    data["is_city"] = data["featurecode"].str.contains("PPL")*1
    data["is_city"] = data["is_city"].fillna(0)

    #add admin tag: 0:high --> 6:low
    data['adm_level'] = data["featurecode"].str.findall('\d+').str.join("").replace("", 6)
    data['adm_level'] = data["adm_level"]*(1-data["is_capital"])*(1-data["is_country"])
    data.loc[data.featurecode == "PPLA", "adm_level"] = 1
    data["adm_level"] = data["adm_level"].fillna(6)
    data.adm_level = data.adm_level.astype(int)
    
    #initial prediction
    data['predict'] = sig(np.dot(np.array(data[["is_pop","adm_level"]]), coeffs[1:3]) + coeffs[0])

    # compute pairwise distance between candidates
    logging.info("compute pairwise distance ---------")
    sqlcode = '''
    select a.lead_id, a.match name1, a.geonameid geonameid1, a.latitude latitude1, a.longitude longitude1,
        b.match name2, b.geonameid geonameid2, b.latitude latitude2, b.longitude longitude2 
    from data as a
    join data as b
    on a.lead_id = b.lead_id and a.match <> b.match'''

    pairs = ps.sqldf(sqlcode, locals())
    pairs['coordinate1'] = list(zip(pairs.latitude1, pairs.longitude1))
    pairs['coordinate2'] = list(zip(pairs.latitude2, pairs.longitude2))

    pairs["distance"] = pairs.apply(lambda x: great_circle(x.coordinate1, x.coordinate2), axis = 1)
    pairs.drop(columns = ['coordinate1', 'coordinate2'], inplace = True)

    for i in range(iters):   

        #is highest ranked candidate so far?
        sqlcode = '''
                select *,
                    case when best == 1 then 1
                         when best <> 1 then 0
                    end as isbest
                from(
                select *, rank() over(partition by lead_id, match order by predict desc) best from data) a
                '''    
        data = ps.sqldf(sqlcode, locals())

        #country distribution (weighted)
        sqlcode = '''
        select * from (
        select lead_id, match, countrycode, predict,
        rank() over (partition by lead_id, match, countrycode order by target desc) rankn
        from data)a
        where rankn = 1
        '''
        country_distr = ps.sqldf(sqlcode, locals())

        sqlcode = '''
        select lead_id, countrycode, sum(sumw) sumw
        from(
        select lead_id, match, countrycode, predict / sum(predict) over(partition by lead_id, match) sumw
        from country_distr)a
        group by lead_id, countrycode
        order by lead_id, sum(sumw) desc'''
        country_distr = ps.sqldf(sqlcode, locals())
        country_distr.sumw = country_distr.groupby("lead_id")['sumw'].transform(lambda x: (x - x.mean()) / x.std())

        sqlcode = '''
        select a.*, b.sumw
        from data a
        join country_distr b on a.lead_id = b.lead_id and a.countrycode = b.countrycode'''

        data = ps.sqldf(sqlcode, locals())

        #average distance from other candidates (weighted)
        sqlcode = '''
        select m2.lead_id, m2.name1, m2.geonameid1, avg(m2.wdistance) wdistance
        from
        (
        select m.lead_id, m.name1, m.name2, m.geonameid1, sum(m.distance*m.distance*m.weight)/sum(m.weight) wdistance
        from (
            select a.lead_id, a.name1, a.name2, a.geonameid1, a.geonameid2, a.distance, b.predict weight
            from pairs a
            join data b on a.lead_id = b.lead_id and a.geonameid2 = b.geonameid
            ) m
        group by m.lead_id, m.name1, m.name2, m.geonameid1
        ) m2
        group by
        m2.lead_id, m2.name1, m2.geonameid1
        '''

        wdistance = ps.sqldf(sqlcode, locals())

        sqlcode = '''
        select a.lead_id, a.match, a.geonameid, latitude, longitude,
           population, featurecode, target, is_country, is_pop,
           is_capital, adm_level, is_city, predict, sumw, countrycode, isbest, test_set,
           b.wdistance 
           from data a
        join wdistance b on a.lead_id = b.lead_id and a.geonameid = b.geonameid1 and a.match = b.name1'''

        data= ps.sqldf(sqlcode, locals())
        data["wdistance"] = data.groupby('lead_id')["wdistance"].transform(lambda x: (x - x.mean()) / x.std())

        #is closest candidate so far?
        sqlcode = '''
                select *,
                    case when closest == 1 then 1
                         when closest <> 1 then 0
                    end as isclosest
                from(
                select *, row_number() over(partition by lead_id, match order by wdistance) closest from data) a
                '''    
        data = ps.sqldf(sqlcode, locals())

        # compute regression with new features
        X = np.array(data[["is_pop","adm_level", "wdistance", "sumw", "isclosest", "isbest"]]) 

        data['predict'] = sig(np.dot(X, coeffs[1:]) + coeffs[0])
        logging.info('iter', i+1)
    return data

def ThresholdDisambiguation(data, threshold):
    sqlcode = '''
    select lead_id, match, geonameid, latitude, longitude, population,
           featurecode, target, is_country, is_pop, is_capital,
           adm_level, is_city, predict, sumw, countrycode, isbest, isclosest,
           test_set, wdistance,
        case when rown = 1 and predict > {threshold} then 1.0 else 0.0 end as final_pred
    from (
    select *, row_number() over (partition by lead_id, match order by predict desc) rown
    from data
    ) a'''.format(threshold = threshold)
    
    return ps.sqldf(sqlcode, locals())

"""
def PlotEvalDistance(target,result):
    sqlcode = '''
    select distinct a.lead_id, a.match, a.is_toponym, b.geonameid geonameid_true, b.latitude lat_true, b.longitude lon_true,
        c.geonameid geonameid_pred, c.latitude lat_pred, c.longitude lon_pred
    from (select distinct lead_id, match, is_toponym from target) a
    left join (select * from target where target = 1) b on a.lead_id = b.lead_id and a.match = b.match
    left join (select * from result where final_pred = 1) c on a.lead_id = c.lead_id and a.match = c.match
    '''
    match = ps.sqldf(sqlcode, locals())
    match['coord_true'] = list(zip(match.lat_true, match.lon_true))
    match['coord_pred'] = list(zip(match.lat_pred, match.lon_pred))

    match["dist"] = match.apply(lambda x: great_circle(x.coord_true, x.coord_pred), axis = 1)
    
    no_match = match[match.is_toponym == 0].copy()
    match = match[match.is_toponym == 1]
    print("### NON LOCATIONS:", len(no_match))
    print("\t- of which pred correctly ignored:",len(no_match[(no_match.geonameid_true.isnull() & no_match.geonameid_pred.isnull())]))
    print("\t- of which location wrongly predicted:",len(no_match[(no_match.geonameid_true.isnull() & no_match.geonameid_pred.notnull())]))
    print("### LOCATIONS:", len(match))
    print("\texact:",len(match[(match.geonameid_true == match.geonameid_pred) | (match.geonameid_true.isnull() & match.geonameid_pred.isnull())]))
    print("\t- of which match exists and correctly predicted:",len(match[(match.geonameid_true == match.geonameid_pred)]))
    print("\t- of which match does not exists and is correctly ignored:",len(match[(match.geonameid_true.isnull() & match.geonameid_pred.isnull())]))

    print("\twrong:", len(match[(match.geonameid_true != match.geonameid_pred) & (match.geonameid_pred.notnull() | match.geonameid_true.notnull())]))
    print("\t- of which match does not exists, but some value is wrongly predicted:", len(match[(match.geonameid_true.isnull() & match.geonameid_pred.notnull())]))
    print("\t- of which match exists and is not selected (i.e excluded by threshold):", len(match[(match.geonameid_true.notnull() & match.geonameid_pred.isnull())]))
    print("\t- of which wrong candidate:", len(match[(match.geonameid_true != match.geonameid_pred) & (match.dist.notnull())]))
    print("--------------")
    print("total:", len(match)+ len(no_match))

    dist = match[(match.geonameid_true != match.geonameid_pred) & (match.dist.notnull())]["dist"]
    plt.hist(dist, bins = 30)
    plt.axvline(dist.mean(), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(dist.median(), color='r', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    min_xlim, max_xlim = plt.xlim()
    plt.text(dist.median()*1.1, max_ylim*0.5, 'Median: {:.2f}'.format(dist.median()), color = 'r')
    plt.text(dist.mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(dist.mean()))
    plt.text(dist.mean()*1.1, max_ylim*0.83, 'Std: {:.2f}'.format(dist.std()))
    plt.text(max_xlim*0.8, max_ylim*0.94, 'N° ents: {}'.format(len(dist)))
    plt.title("distance distribution for wrong matches")
    plt.show();
    
    return

"""