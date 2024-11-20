import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


driver = webdriver.Chrome("../ChromeDriver/chromedriver_linux64/chromedriver")

driver.get("https://www.google.com/search?gs_ssp=eJzj4tVP1zc0TDKrSDLPK8w1YLRSNagwMTNNtUgxNDdPtrQwT061tDIAyqaZJlskWqaYJadZpBoYegknZ-QXgLBCanF2Zk5xSWleIgA6mxd3&q=chopchop+eskilstuna&oq=chopchop+eskilstuna&aqs=chrome.1.69i59j46i39i175i199j69i59j0i512j0i22i30j69i61j69i60l2.2718j0j4&sourceid=chrome&ie=UTF-8#lrd=0x465e8d177c987ce9:0xb7f5c8a9d6cf8e01,1,,,")


soup = BeautifulSoup(driver.page_source, "html.parser")

#driver.find_elements(By.CLASS_NAME, "AxAp9e xaNsfc ZkkK1e yUTMj k1U36b")[1].click()
time.sleep(10)
wait = WebDriverWait(driver,20)

soup = BeautifulSoup(driver.page_source, "html.parser")

reviews = soup.find_all("div",{"class":"Jtu6Td"})
persons = soup.find_all("div", {"class":"dehysf lTi8oc"})

print(len())
"""
x=0
desiredReviewsCount=30
wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME,"Jtu6Td")))
while x<desiredReviewsCount:
    w = driver.find_elements(By.CLASS_NAME, "lcorif fp-w")
    print(w)
    w.location_once_scrolled_into_view
    x = len(driver.find_element("Jtu6Td"))
"""
#print(len(driver.find_elements_by_xpath("//div[@class='gws-localreviews__general-reviews-block']//div[@class='WMbnJf gws-localreviews__google-review']")))

#time.sleep(3)
#body = driver.find_element(By.CLASS_NAME, "review-dialog-list")
#for i in range(10):
#    body.send_keys(Keys.PAGE_DOWN)

#driver.execute_script('document.querySelector(\'[class="keynav-mode-off screen-mode"]\').scrollBy(0,200)')
#body = r.find("body", {"class":"keynav-mode-off screen-mode"})
#driver.find_element_by_xpath("/html/body/div[3]/div[9]/div[8]/div/div[1]/div/div/div[2]/div[1]/div[1]/div[2]/div/div[1]/span[1]/span/span[1]/span[2]/span[1]/button").click()
#time.sleep(2)

"""driver.find_element_by_class_name("Yr7JMd-pane-hSRGPd").click()
time.sleep(3)
body = driver.find_element_by_tag_name("body")
body.send_keys(Keys.PAGE_DOWN)
print("here")
#button = driver.find_element_by_xpath("/html/body/div[3]/div[9]/div[8]/div/div[1]/div/div/div[43]/div/button/span/span")
#button.click()

time.sleep(2)

x = r.find_all("div", {"class":"ODSEW-ShBeI-ShBeI-content"})
print(x)
"""
#driver.close()