from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys


def waitPageLoaded(pdriver, plogoClassname):
    '''
    Vì các trang product landing page dc thiết kế dưới dạng dynamic site (tức
    data dc load lên dựa vào các javascript event khi người dùng có sự tương
    tác), và các comment chỉ hiển thị khi user lăn chuột đến phần comment, nên
    hàm này giúp giả lập làm điều này.
    '''
    try:
        '''Trong 5s, đợi logo shopee hiện lên'''
        WebDriverWait(pdriver, 5).until(EC.presence_of_element_located(
            (By.CLASS_NAME, plogoClassname)))
        
        old_height = 0
        new_height = pdriver.execute_script("return document.body.scrollHeight")
        while True:
            '''Scroll đến cuối trang'''
            pdriver.execute_script(f"window.scrollTo(0, {new_height});")
            
            old_height = new_height
            new_height = pdriver.execute_script("return document.body.scrollHeight")
            
            if old_height == new_height: # tìm thấy footer
                print(new_height)
                return True
    except TimeoutException:
        '''Trang load ko thành công'''
        return False

def waitPageLoaded2(pdriver, pcssSelector1, pcssSelector2):
    '''
    Vì các trang product landing page dc thiết kế dưới dạng dynamic site (tức
    data dc load lên dựa vào các javascript event khi người dùng có sự tương
    tác), và các comment chỉ hiển thị khi user lăn chuột đến phần comment, nên
    hàm này giúp giả lập làm điều này.
    '''
    try:
        '''Trong 5s, đợi logo shopee hiện lên'''
        WebDriverWait(pdriver, 5).until(EC.presence_of_element_located(
            (By.CLASS_NAME, pcssSelector1)))
        
        old_height = 0
        new_height = pdriver.execute_script("return document.body.scrollHeight")
        while True:
            '''Scroll đến cuối trang'''
            pdriver.execute_script(f"window.scrollTo(0, {new_height});")
            
            if WebDriverWait(pdriver, 2).until(EC.visibility_of_any_elements_located((By.CSS_SELECTOR, pcssSelector2))):
                print("Founded!!!")
                return True
            
            old_height = new_height
            new_height = pdriver.execute_script("return document.body.scrollHeight")
            
            if old_height == new_height: # tìm thấy footer
                print(new_height)
                return True
    except TimeoutException:
        '''Trang load ko thành công'''
        return False


def getProductURLs(purl: str, prange: tuple, pcssSelector: str):
    """
    Hàm này dùng để lấy tất cả các product's urls thỏa pcssSelector

    Args:
        purl (str): trang chính của nhóm mặt hàng cần lấy
        prange (tuple): một tuple (a, b) là số trang chứa các gallary product của shopee
        pcssSelector (str): CSS selector

    Returns:
        (list[str]): chuổi các url của các product
    """
    '''Chọn driver để crawl data là Firefox'''
    driver = webdriver.Firefox()
    
    '''Dùng để lưu các hyperlink đến các product's landing page'''
    product_urls = []
    
    '''Đi từ trang 0 đến 99 (UI là 1 đến 100)'''
    for i in range(prange[0], prange[1]):
        url = f"{purl}{i}" # access vào trang thứ i
        driver.get(url) # mở Firefox để access vào `url`
        
        '''Nếu đi đến cuối trang thành công'''
        if waitPageLoaded(driver, "header-with-search__logo-section"):
            '''Lấy các href values từ tất cả thẻ anchor thỏa `pcssSelector`'''
            new_product_urls = [anchor.get_attribute('href') for anchor in driver.find_elements_by_css_selector(pcssSelector)]
            product_urls += new_product_urls # thêm vào kết quả trả về
            
    driver.close()
    driver.quit()
    
    return product_urls
            

class Review:
    def __init__(self, pcomment: str, prating: int):
        """
        Constructor
        
        Args:
            pcomment (str): comment
            prating (int): rating nằm trong khoảng [1, 5]
        """
        self.icomment = pcomment
        self.irating = prating       
    
    
def getProductReviews(pproductURL: str):
    driver = webdriver.Firefox()
    product_reviews = []
    locator_button_focus = (By.CLASS_NAME, "shopee-button-solid--primary")
    locator_button = (By.CLASS_NAME, "shopee-icon-button--right")
    buttons_box = {}
    
    driver.get(pproductURL)
    
    if waitPageLoaded2(driver, "_35Jk8Z", "div.shopee-product-rating"):
        while True:
            reviews = driver.find_elements_by_css_selector("div.shopee-product-rating")
            
            if type(reviews) == list and len(reviews) != 0:                        
                for review in reviews:
                    comment = review.find_element_by_class_name("shopee-product-rating__content").text
                    rating = len(review.find_elements_by_class_name("icon-rating-solid--active"))
                    
                    new_review = Review(comment, rating)
                    product_reviews.append(new_review)
                    
            try:
                button_next = WebDriverWait(driver, 2).until(EC.element_to_be_clickable(locator_button))    
                if button_next is not None: button_next.click()
                
                button_focus = WebDriverWait(driver, 2).until(EC.presence_of_element_located(locator_button_focus))
                if button_focus is not None:
                    key = button_focus.text.strip()
                    buttons_box[key] = buttons_box.get(key, 0) + 1
                    
                    if buttons_box.get(key) > 1:
                        break
            except TimeoutException:
                break
    
    driver.close()
    driver.quit()
    
    return product_reviews