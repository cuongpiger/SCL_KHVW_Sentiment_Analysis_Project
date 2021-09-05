import os
import re
import pandas as pd
import unicodedata

from os.path import isfile, join
from typing import List, Dict
import modules.user_object_defined as udt

def getAllFolderPath(ppath: str) -> (List[str]):
    """
    Hàm dùng để lấy tất cả các thư mục con của `ppath`

    Args:
        ppath (str): đường dẫn thư mục cha

    Returns:
        List[str]
    """
    return [re.sub("/+", "/", f"{ppath}/{name}/") for name in os.listdir(ppath)]
    
    
def getFilenames(pdirectoryPath: str) -> (List[str]):
    """
    Dùng để lấy tất các các filename nằm trong `pdirectoryPath`

    Args:
        pdirectoryPath (str): đường dẫn của thư mục cha

    Returns:
        List[str]: 
    """
    return [f for f in os.listdir(pdirectoryPath) if isfile(join(pdirectoryPath, f))]
    
    
def readReviews(pdirectoryPath: List[str]) -> (udt.Dataframe):
    """
    Dùng để đọc tất cả các review từ các file csv

    Args:
        pdirectoryPath (List[str]): đường dẫn của folder chứa các file csv

    Returns:
        [udt.Dataframe]: pandas dataframe gồm 2 feature là `raw_comment` và `rating`
    """
    reviews = pd.DataFrame(columns=['comment', 'rating'])
    
    for path in pdirectoryPath:
        csv_paths = getFilenames(path) # lấy tất cả các file csv nằm trong path
        
        for csv_path in csv_paths:
            df = pd.read_csv(path + csv_path) # đọc file csv mới lên
            reviews = pd.concat([reviews, df], axis=0) # nối df với review
            
    '''Đổi tên cột comment thành raw_comment và reset index cho toàn bộ dataframe'''        
    return reviews.rename(columns={"comment":"raw_comment"}).reset_index(drop=True)


def labelRating(pdataframe: udt.Dataframe, pthreshold:int = 4) -> (udt.Dataframe):
    """
    Phân nhóm các review thành 2 nhóm là negative và positive dựa nào `pthreshold`

    Args:
        pdataframe (udt.Dataframe): các reviews
        pthreshold (int, optional): ngưỡng phân nhóm. Defaults to 4.

    Returns:
        [udt.Dataframe]: dataframe mới copy từ `pdataframe` nhưng được thêm feature `label`
    """
    pdataframe['label'] = pdataframe['rating'].apply(lambda rt: 1 if rt >= pthreshold else 0)
    return pdataframe


def buildDictionaryFromFile(ppath: str) -> (Dict[str, str]):
    """
    Dùng để xây dựng một từ điển tử filepath

    Args:
        ppath (str): đường dẫn file

    Returns:
        [Dict[str, str]]: 
    """
    d = {}
    
    with open(ppath) as rows:
        for row in rows:
            prefix, suffix = row.strip().split(',')
            prefix = unicodedata.normalize('NFD', prefix)
            suffix = unicodedata.normalize('NFD', suffix)
            d[prefix] = suffix
            
    return d
            