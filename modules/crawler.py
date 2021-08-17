from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys


    


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
    def waitPageLoaded(pdriver):
        '''
        Vì các trang product landing page dc thiết kế dưới dạng dynamic site (tức
        data dc load lên dựa vào các javascript event khi người dùng có sự tương
        tác), và các comment chỉ hiển thị khi user lăn chuột đến phần comment, nên
        hàm này giúp giả lập làm điều này.
        '''
        try:
            '''Trong 5s, đợi logo shopee hiện lên'''
            WebDriverWait(pdriver, 5).until(EC.presence_of_element_located(
                (By.CLASS_NAME, "header-with-search__logo-section")))
            
            old_height = 0
            new_height = pdriver.execute_script("return document.body.scrollHeight")
            while True:
                '''Scroll đến cuối trang'''
                pdriver.execute_script(
                    f"window.scroolTo(0, {new_height}")
                
                old_height = new_height
                new_height = pdriver.execute_script("return document.body.scrollHeight")
                
                if old_height == new_height: # tìm thấy footer
                    return True
        except TimeoutException:
            '''Trang load ko thành công'''
            return False
        
            
    
    '''Chọn driver để crawl data là Firefox'''
    driver = webdriver.Firefox()
    
    '''Dùng để lưu các hyperlink đến các product's landing page'''
    product_urls = []
    
    '''Đi từ trang 0 đến 99 (UI là 1 đến 100)'''
    for i in range(prange[0], prange[1]):
        url = f"{purl}{i}" # access vào trang thứ i
        driver.get(url) # mở Firefox để access vào `url`
        
        '''Nếu đi đến cuối trang thành công'''
        if waitPageLoaded(driver):
            '''Lấy các href values từ tất cả thẻ anchor thỏa `pcssSelector`'''
            new_product_urls = [
                anchor.get_attribute('href') for anchor in 
                driver.find_element_by_css_selector(pcssSelector)]
            product_urls += new_product_urls # thêm vào kết quả trả về
            
    driver.close()
    driver.quit()
    
    return product_urls
            
            
    