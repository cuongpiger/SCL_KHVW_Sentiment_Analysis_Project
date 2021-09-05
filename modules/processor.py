import re
import emojis
import unicodedata

from typing import List, Dict
import modules.regex_patterns as RegPattern
import modules.user_object_defined as udt

def containsURL(ptext: str) -> (int):
    """
    Dùng kiểm tra ptext có chứa url hay ko

    Args:
        ptext (str): comment

    Returns:
        [int]: 1 có 0 ko
    """
    flag = re.search(RegPattern.URL, ptext)
    return int(flag is not None)


def containAdvertisement(ptext: str) -> (int):
    """
    Lọc các ptext mà upper letter chiếm hơn 1 nữa độ dài chuổi

    Args:
        ptext (str): review

    Returns:
        [int]: 1 nếu upper letter chiếm hơn 1 nữa độ dài chuổi, otherwise 0
    """
    text_no_upper = re.sub(RegPattern.UTF8_UPPER, '', ptext) # thay các upper letter thành ''
    return int(len(text_no_upper) / len(ptext) >= 0.5)


def extractEmoji(ptext: str) -> (str):
    """
    Trích xuất emoji từ comment

    Args:
        ptext (str): comment

    Returns:
        [str]: string chứa các emoji
    """
    return ' '.join(list(emojis.get(ptext)))


def normalizeComment(ptext: str, plower: bool = True) -> (str):
    """
    Chuẩn hóa text bằng cách lower nó sau đó sử dụng phương pháp NFD để biểu diễn text

    Args:
        ptext (str): comment
        plower (bool): có lower ko

    Returns:
        [str]: comment đã lower và chuẩn hóa
    """
    ptext = ptext.lower() if plower else ptext
    return unicodedata.normalize('NFD', ptext)


def removeSpecialLetters(ptext: str) -> (str):
    """
    Dùng xóa các kí tự đặc biệt

    Args:
        ptext (str): comment

    Returns:
        [str]: comment without special characters
    """
    return re.sub("\s+", " ", re.sub(RegPattern.UTF8_LOWER, " ", ptext)).strip() 


def removeDuplicateLetters(ptext: str) -> (str):
    """
    Hàm dùng xóa các kí tự bị dupplucate, giả sử :
      * ptext = 'okkkkkkkkkkkkkkkkkkkkkk chờiiiiiiiiii ơiiiiiiii xinhhhhhhhhhhhh quá đẹppppppppp xỉuuuuuuu'
      * Sau khi dùng hàm này thì thành:
        ptext = 'ok chời ơi xinh quá đẹp xỉu'
      
    Args:
        ptext (str): comment

    Returns:
        [str]: comment that removing duplicated letters 
    """
    return re.sub(r'(.)\1+', r'\1', ptext)


def replaceWithDictionary(ptext: str, pdictionary: Dict[str, str]) -> (str):
    """
    Hàm này dùng để thay thế các từ đơn trong ptext mà là key của pdictionary, sau đó
    thay thế từ này bằng value tương ứng với key đó.

    Args:
        ptext (str): comment
        pdictionary (Dict[str, str]): dictionary

    Returns:
        (str): comment đã dc thay thế bởi các value match với pdictionary
    """
    words = ptext.strip().split(' ')
    new_words = []
    
    for word in words:
        word = pdictionary.get(word, word)
        new_words.append(word)
        
    return ' '.join(new_words)


def printAfterProcess(pdataframe: udt.Dataframe, pcolumnName: str = 'label'):
    """
    Dùng để in các giá trị về shape, số lượng các sample của từng nhóm mỗi khi ta chỉnh
    sửa dataframe

    Args:
        pdataframe (udt.Dataframe): các reviews
        pcolumnName (str): cột cần value_count
    """
    print(f"Shape: {pdataframe.shape}")
    print(pdataframe[pcolumnName].value_counts())
    