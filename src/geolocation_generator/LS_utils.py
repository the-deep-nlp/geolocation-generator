import json
import pandas as pd
from os import path
import pandasql as ps


def build_LS_json(df, text, doc_id, model_version, max_str_len):
    predictions = [build_LS_predictions(df, doc_id, model_version)]
    ####
    if max_str_len:
        #shorten text
        next_sep = text[max_str_len:].find('\n\n')

        #round to next separator
        if next_sep>0:
            max_str_len += next_sep
        else: 
            max_str_len = 10**6

        #crop text and predictions
        text = text[:max_str_len]
        predictions = [elem for elem in predictions[0]['result'] if (elem['value']['end']<max_str_len)]

    ####

    data = {"text": text}

    file = {"id": doc_id, "data":data, "predictions":predictions}

    return file

    ####

def build_LS_predictions(df, doc_id, model_version):
    #get offsets of tags in the text
    offsets = df[["locs","offset", "type"]].explode("locs")
    offsets = offsets[~pd.isna(offsets["locs"])]
    offsets = [[x["offset_start"] + y, x["offset_end"] + y, z]  for x, y, z in zip(offsets['locs'], offsets['offset'], offsets['type'])]

    result = []
    for offset in offsets:
        label = {"value": {"start": offset[0],"end": offset[1],"labels": [offset[2]]},
                "id": '-'.join([str(doc_id),str(offset[0]), str(offset[1])]),
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "manual"
                }
        result.append(label)
    
    return {"model_version":model_version, "result": result}

def write_LS_json(df, text, doc_id, model_version = "0", out_folder = "LS_predictions", append = False, max_str_len = None):
    if not append:
        file = build_LS_json(df, text, doc_id, model_version, max_str_len)
        with open(path.join(out_folder,str(doc_id)+'.json'), 'w') as f:
            json.dump(file, f, indent = 2)
        print("Predictions created:", str(model_version), "to", str(doc_id)+".json")
    else:
        file_exists = path.isfile(path.join("LS_predictions",str(doc_id)+'.json'))
        if file_exists:
            with open(path.join("LS_predictions",str(doc_id)+'.json'), 'r') as f:
                file = json.load(f)
            file["predictions"] = [pred for pred in file['predictions'] if pred['model_version'] != model_version]
            file["predictions"].append(build_LS_predictions(df, doc_id, model_version))

            with open(path.join(out_folder,str(doc_id)+'.json'), 'w') as f:
                json.dump(file, f, indent = 2)
            print("Predictions appended:", str(model_version), "to", str(doc_id)+".json")

        else:
            append = False
            write_LS_json(df, text, doc_id, model_version, out_folder, append)

def merge_overlapping_models(df1, df2):
    df1_ = df1.copy()
    df = df1.merge(df2, on = ["page", "block"], suffixes=('_x', '_y'))[['page', 'block', 'text_x', 'locs_x', 'locs_y']]
    df['locs_xy'] = df["locs_x"] + df["locs_y"]
    df['locs_xy'] = df['locs_xy'].apply(lambda x: pd.DataFrame(x).drop_duplicates().to_dict('records'))
    df1_['locs'] = df.apply(lambda x: merge_overlapping_ents(x.locs_xy, x.text_x), axis = 1)
    df1_['locs'] = df1_.apply(lambda x: merge_overlapping_ents(x.locs, x.text), axis = 1)
    df1_['type'] = 'LOC'
    
    return df1_
    
def merge_overlapping_ents(locs, text):
    if len(locs) > 0:
        locs_to_df = pd.DataFrame(locs)
        locs_to_df = locs_to_df.merge(locs_to_df, how = 'cross')[['offset_start_x', 'offset_end_x', 'offset_start_y', 'offset_end_y']]
        merged = locs_to_df.loc[(locs_to_df.offset_start_x <= locs_to_df.offset_end_y) \
                                & (locs_to_df.offset_start_y <= locs_to_df.offset_end_x),:].reset_index(drop = True)
        merged = merged.groupby(["offset_start_x", "offset_end_x"]).agg(offset_start = ('offset_start_y','min'),
                                                                        offset_end = ('offset_end_y','max')
                                                                       ).drop_duplicates().sort_values(by=['offset_start_x']
                                                                                                      ).reset_index(drop = True)
    
        out = [{'ent':text[off[0]:off[1]],'offset_start':off[0], 'offset_end':off[1]} for off in merged.values.tolist()]
    else: 
        out = []
    
    return out

def explode_locs(df):
    df_c = df.copy()
    df_c.loc[:,'sentence'] = df_c.index
    _df = df_c.loc[df_c.loc[:,'locs'].astype(bool),:].explode('locs')
    return pd.concat([_df.loc[:, _df.columns != 'locs'], _df.loc[:,'locs'].apply(pd.Series)], axis=1)

def separate_models(df1,df2, df2_type):
    df1_e = explode_locs(df1)
    df2_e = explode_locs(df2)
    
    if len(df1_e)>0:
        sqlcode = '''
            select a.*
            from df2_e a
            left join df1_e b
            on a.page = b.page and a.block = b.block and a.offset_start<= b.offset_end and b.offset_start <= a.offset_end
            where b.page is null
            '''
        exclude = ps.sqldf(sqlcode,locals())
    else: 
        exclude = df2_e
    if len(exclude) > 0:
        exclude = exclude.groupby(['page','block'])[['ent','offset_start','offset_end']].apply(lambda x:
                                                                                               pd.Series({'locs_y':x.to_dict('records')}))
        exclude = exclude.reset_index()

        df = df2.merge(exclude, how = 'left',on = ['page','block'])
        df.loc[df.locs_y.isnull(), 'locs_y'] = df.loc[df.locs_y.isnull(), 'locs_y'].apply(lambda x: [])
        df['diff'] = df[['locs','locs_y']].apply(lambda x: [i for i in x[0] if i not in x[1]], axis = 1)

        tmp1 = df[['page', 'block', 'text', 'offset', 'sentence', 'diff', 'type']].rename(columns = {'diff':'locs'})
        tmp2 = df[['page', 'block', 'text', 'offset', 'sentence', 'locs_y', 'type']].rename(columns = {'locs_y':'locs'})
        tmp2.type = df2_type
        out = pd.concat([tmp1,tmp2])
    else:
        out = df2
    return out
