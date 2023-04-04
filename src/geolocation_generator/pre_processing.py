import pandas as pd
import re
from os import path

def fetch_text(folder_path, lead_id, max_pages):
    with open (path.join(folder_path, str(lead_id)+'.txt'), 'r', encoding='UTF-8') as f:
        text = f.read()
    if '********* [PAGE ' +str(max_pages)+' END] *********' in text:
        text = ''.join(text.rpartition('********* [PAGE ' +str(max_pages)+' END] *********\n')[0:2])
    text = re.sub(f'\*\*\*\*\*\*\*\*\* \[', '[', text)
    text = re.sub(f'\] \*\*\*\*\*\*\*\*\*\\n', ']\\n', text)
    text = re.sub(f'\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\\n', '\t\n', text )
    text = str(lead_id)+'\n'+text
    return text

def offset_find(text, page, block):
    p_start = re.search(f'\[PAGE {page} START\]\\n', text).start()
    p_end = re.search(f'\[PAGE {page} END\]\\n', text).start()

    sep = '\\t\\n'
    count = 0 
    for match in re.finditer(sep, text[p_start:p_end]):
        count += 1
        if block == count:
            offset = p_start + match.end()
        else: 
            pass
    return offset

def pre_processing(folder_path, lead_id, max_pages = 5):
    #fetch text 
    text = fetch_text(folder_path, lead_id, max_pages)
    
    #split text in pages a blocks
    page_start_pattern = '\[PAGE [0-9]+ START\]\\n'
    page_end_pattern = '\*\*\*\*\*\*\*\*\* \[PAGE [0-9]+ END\]\\n'
    data = re.sub(page_end_pattern, '', text)
    data = re.split(page_start_pattern, data)
    del data[0]
    data = [(idx + 1, item) for idx,item in enumerate(data)]
    data = [[page[0],page[1].split('\t\n')[1:]] for page in data]
    data = [[page[0], block + 1, text]for page in data for block, text in enumerate(page[1])]
    df = pd.DataFrame(data, columns= ["page", "block", "text"])

    #add offset
    df["offset"] = df.apply(lambda row: offset_find(text, row["page"], row["block"]), axis= 1)

    #clean data frame
    df.text = df.text.str.rstrip("\n")
    #df = df[df.text.str.len()>50].reset_index(drop = True)

    return df, text