import re
import emojis
import unicodedata

from typing import List
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
    