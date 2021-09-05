import unicodedata


'''Pattern dùng để kiểm tra text có chứa url hay ko'''
URL = unicodedata.normalize('NFD', r"(?i)\b((?:http[s{0,1}]?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")


'''Các kí tự utf-8 upper'''
UTF8_UPPER =  unicodedata.normalize('NFD', r"[^A-ZÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]")


'''Các kí tự utf-8 lower'''
UTF8_LOWER = unicodedata.normalize('NFD', r"[^a-záàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ ]")
