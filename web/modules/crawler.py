from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import pandas as pd
import re
import requests

from typing import List, Tuple

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
    
        '''Chiều cao cũ của trang'''    
        old_height = 0

        '''Chiều cao mà trang có thể lăn đến ở thời điểm hiện tại'''
        new_height = pdriver.execute_script("return document.body.scrollHeight")
        while True:
            '''Scroll đến chiều cao mà trang đã load dc'''
            pdriver.execute_script(f"window.scrollTo(0, {new_height});")
            
            '''Nếu đã load đến cuối trang thì old_height sẽ bằng new_height'''
            old_height = new_height
            new_height = pdriver.execute_script("return document.body.scrollHeight")
            
            if old_height == new_height: # tìm thấy footer
                return True
    except TimeoutException:
        '''Trang load ko thành công do quá timeout 5s'''
        return False

def waitPageLoaded2(pdriver, pcssSelector1, pcssSelector2):
    """
    Hàm này cũng dùng để scroll trang nhưng trong quá trình scroll sẽ kiểm tra
    xem có css selector nào match với pcssSelector2 ko, nếu có thì dừng lại, còn
    ko thì tiếp tục scroll tiếp cho đến khi ko còn scroll dc nữa

    Returns:
        (bool): True nếu tìm thấy pcssSelector2, hoặc đến cuối trang, False nếu
            cả hai điều kiện trên đều ko tìm dc
    """
    try:
        '''Trong 5s, đợi pcssSelector1 hiện lên thành công, ở đây là hình sản phẩm'''
        WebDriverWait(pdriver, 5).until(EC.presence_of_element_located(
            (By.CLASS_NAME, pcssSelector1)))
        
        '''Phần này tương tự như hàm waitLoadLoaded'''
        old_height = 0
        new_height = pdriver.execute_script("return document.body.scrollHeight")
        while True:
            '''Scroll đến cuối trang'''
            pdriver.execute_script(f"window.scrollTo(0, {new_height});")
            
            try:
                '''Nếu tìm thấy pcssSelector2 thì trả về True'''
                if WebDriverWait(pdriver, 2).until(EC.visibility_of_any_elements_located((By.CSS_SELECTOR, pcssSelector2))):
                    return True
            except TimeoutException:
                '''Do quá timeout 2s, nên rơi vào đây, có nghĩa chưa tìm thấy nên chạy tiếp'''
                pass
            
            old_height = new_height
            new_height = pdriver.execute_script("return document.body.scrollHeight")
            
            '''Đã đến cuối trang'''
            if old_height == new_height:
                return True
    except TimeoutException:
        '''Trang load ko thành công do quá timeout 5s'''
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
        
    '''Đóng Firefox'''    
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
    product_reviews = [] # chứa các review object
    locator_button_focus = (By.CLASS_NAME, "shopee-button-solid--primary") # button của selected navigation page
    locator_button = (By.CLASS_NAME, "shopee-icon-button--right") # button next navigation page 
    buttons_box = {} # kiểm tra các page nào đã duyệt qua rồi
    
    driver.get(pproductURL)
    
    '''Nếu hình ảnh sản phẩm đã load lên thành công'''
    if waitPageLoaded2(driver, "_35Jk8Z", "div.shopee-product-rating"):
        while True:
            '''Tìm thẻ div chứa các comment thỏa argument css selector'''
            reviews = driver.find_elements_by_css_selector("div.shopee-product-rating")
            
            '''Nếu kết quả trả về có dạng list và non empty'''
            if type(reviews) == list and len(reviews) != 0:                        
                '''Tạo các object Review và thể vào list kết quả'''
                for review in reviews:
                    '''Trong parent html element, tìm các thẻ con thỏa argument css selector, đây là các comment'''
                    comment = review.find_element_by_class_name("shopee-product-rating__content").text
                    
                    '''Tìm số icon ngôi sao được tô màu, lấy len là số sao mà khách hàng đánh giá cho sản phẩm'''
                    rating = len(review.find_elements_by_class_name("icon-rating-solid--active"))
                    
                    '''Tạo một new Review object vào cho vào mảng kết quả trả về'''
                    new_review = Review(comment, rating)
                    product_reviews.append(new_review)
                    
            try:
                '''Tìm button next qua navigation tiếp theo, xem có thể click dc ko, nếu dc click vào qua trang tiếp theo'''
                button_next = WebDriverWait(driver, 2).until(EC.element_to_be_clickable(locator_button))    
                if button_next is not None: button_next.click()
                
                '''Lấy giá trị text inner HTML element, thêm nó vào buttons_box'''
                button_focus = WebDriverWait(driver, 2).until(EC.presence_of_element_located(locator_button_focus))
                if button_focus is not None:
                    key = button_focus.text.strip() # strip cho mất khoảng trắng đầu đuôi
                    buttons_box[key] = buttons_box.get(key, 0) + 1 # thêm nó vào dictionary các trang đã đi qua
                    
                    '''Nếu một trang đã đi qua quá 1 lần, thì đã đến cuối trang, dừng crawl sản phẩm này lại'''
                    if buttons_box.get(key) > 1:
                        break
            except TimeoutException:
                break
    
    '''Đóng firefox'''
    driver.close()
    driver.quit()
    
    '''Trả kết quả crawl dc về'''
    return product_reviews


'''
  * Cho requests biết các thông tin về hệ điều hành (Linux, MacOS, Windows), kiến trúc 32 hay 64 bit,
version của Gecko và Firefox (ko cần quan tâm về cái này lắm, người ta đã cấu hình sẵn, tra bảng lụm
vô rồi sai thôi: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/User-Agent/Firefox).
  * Ở đây sài Linux x64.
'''
headers = {
    'User-Agent': "Mozilla/5.0 (X11; Linux x86_64; rv:10.0) Gecko/20100101 Firefox/10.0"
}

def getProductReviewsAPI(pproductURL: str) -> (List[Review]):
    """
    Hàm này dùng để crawl data bằng cách sử dụng API.

    Args:
        pproductURL (str): là URL đến với trang riêng của sản phẫm.
    """
    def collect_comment(ppage: int, pproductID: str, pshopID: str):
        """
        Một hàm bổ trợ dùng để crawl data trên một trang nhất định (navigation)

        Args:
            ppage (int): trang số mấy ta muốn lấy, bắt đầu từ 0
            pproductID (str): mã id của sản phẩm
            pshopID (str): mã id của người bán sản phẩm

        Returns:
            List[Review]:
        """
        
        ''' Các tham số cần truyền vào API để crawl data '''
        params = (
            ('filter', '0'),
            ('flag', '1'),
            ('itemid', pproductID),
            ('limit', '6'), # số 6 ở đây là nếu đếm số comment trong một navigation thì một trang có tối đa là 6 comment
            ('offset', ppage),
            ('shopid', pshopID),
            ('type', '0'))

        ''' Gửi một request đến API và nhận về response '''
        response = requests.get('https://shopee.vn/api/v2/item/get_ratings', headers=headers, params=params)
        return response.json().get('data').get('ratings')
    
    def parse_comment(presponse) -> (Tuple[int, str]):
        """
        Hàm này dùng để parse một response từ API thành hai phần là comment và rating

        Args:
            presponse ([type]): response trả về vừ API

        Returns:
            Tuple[int, str]: lần lượt là rating và comment của review
        """
        ratting = []
        comment = []
        for elem in presponse:
            ratting.append(elem.get('rating_star'))
            comment.append(elem.get('comment'))
        return ratting, comment
        
    
    '''
      Mỗi một URL sản phẩm nếu bạn chú ý sẽ có một thành phần là `i.0284272429.2342427424` như thế này,
    nó dùng để định danh cho sản phẩm này và mã code đầu tiên là id của người bán sản phẩm và mã code
    thứ hai chính là id của sản phẩm. Dưới đây ta dủng regex của python đê lấy hai mã code này.
    '''
    identifier: str  = re.search("i\.\d+.\d+", pproductURL).group(0)
    _, shop_id, product_id = identifier.split('.')
    product_reviews = []

    no_reviews = 0 # đây là trang navigation cần lấy, bắt đầu từ 0
    while True:
        res = collect_comment(no_reviews*6, product_id, shop_id) # trả về response
        rattings, comments = parse_comment(res) # parse nó thành comment và rating
        no_reviews += 1
        
        if not ''.join(comments): break # nếu như comment trả về toàn là empty thì đã hết data để crawl và API đang trả về rác
        
        for comment, ratting in zip(comments, rattings): # bỏ vào kết quả tra về
            if not comment: continue # đụng đến comment rỗng thì ngừng
            
            product_reviews.append(Review(comment, ratting))
            
    return product_reviews



def writeToCsv(pfilePath: str, previews: list):
    """
    Hàm dùng để ghi các review object của một sản phẩm vào file csv

    Args:
        pfilePath (str): nơi lưu tập tin
        previews (list): list các review object của sản phẩm
    """
    if previews == []: return
    reviews = []

    for row in previews:
        try:
            '''Nếu comment rỗng hoặc nội dung bị admin ban thì ignore nó'''
            comment = row.icomment.strip()
            if comment == "" or comment == "****** Đánh giá đã bị ẩn do nội dung không phù hợp. ******": continue

            reviews.append((comment, row.irating))
        except:
            continue

    '''Tạo datafram và ghi ra file'''
    df = pd.DataFrame(reviews, columns=['comment', 'rating'])
    df.to_csv(pfilePath, index=False, header=True)