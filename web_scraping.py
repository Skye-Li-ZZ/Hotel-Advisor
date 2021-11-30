# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 17:09:38 2021

@author: Skye Li
"""

# library to scrape website
from selenium import webdriver
import time
import pandas as pd
from random import randint

# choose the scraper
# choosing your scraper
chosen_driver=['Chrome','Firefox'][0]
if chosen_driver=='Chrome':
    driver    = webdriver.Chrome()
if chosen_driver=='Firefox':
    driver    = webdriver.Firefox()

"""
lists of root urls
"""

# MA
#url = 'https://www.tripadvisor.com/Hotels-g28942-Massachusetts-Hotels.html'
#n_page = 90

# tripadvisor, hotels in Greater Boston
#url = 'https://www.tripadvisor.com/Hotels-g60745-Boston_Massachusetts-Hotels.html'
#n_page = 8

# North Boston
#url = 'https://www.tripadvisor.com/Hotels-g48276-North_Boston_New_York-Hotels.html'

# Greater Merrimack Valley
#url = 'https://www.tripadvisor.com/Hotels-g10515676-Greater_Merrimack_Valley_Massachusetts-Hotels.html'

# Plymouth
#url = 'https://www.tripadvisor.com/Hotels-g41773-Plymouth_Massachusetts-Hotels.html'
#n_page = 1

# Cape Cod
#url = 'https://www.tripadvisor.com/Hotels-g185492-Cape_Cod_Massachusetts-Hotels.html'
#n_page = 17

# Martha'sVineyard
#url = 'https://www.tripadvisor.com/Hotels-g29528-Martha_s_Vineyard_Massachusetts-Hotels.html'
#n_page = 3

# Nantucket
#url = 'https://www.tripadvisor.com/Hotels-g29527-Nantucket_Massachusetts-Hotels.html'
#n_page = 2

# Greater_Springfield
#url = 'https://www.tripadvisor.com/Hotels-g17388906-Greater_Springfield_Massachusetts-Hotels.html'
#n_page=4

# Hampshire_County
#url = 'https://www.tripadvisor.com/Hotels-g4312749-Hampshire_County_Massachusetts-Hotels.html'
#n_page = 2

# Berckshires
#url = 'https://www.tripadvisor.com/Hotels-g659471-Berkshires_Massachusetts-Hotels.html'
#n_page = 6

# Franklin County
#url = 'https://www.tripadvisor.com/Hotels-g41575-Franklin_Massachusetts-Hotels.html'

# Bristol
#url = 'https://www.tripadvisor.com/Hotels-g54063-Bristol_Rhode_Island-Hotels.html'
#n_page = 1

# driver.get(url)

"""
Scrape links from all pages
"""
hotel_links = []
hotel_names = []
hotel_stars = []
hotel_ranks = []
hotel_classes = []
hotel_styles = []
hotel_walkers = []
hotel_restaurant = []
hotel_attraction = []
hotel_reviews_0 = []
hotel_reviews_1 = []
hotel_reviews_2 = []
hotel_reviews_3 = []
hotel_reviews_4 = []
hotel_ratings_0 = []
hotel_ratings_1 = []
hotel_ratings_2 = []
hotel_ratings_3 = []
hotel_ratings_4 = []
hotel_nreviews = []

for i in range(n_page):
    driver.get(url)
    time.sleep(randint(1,3))
    """for each page"""
    # find all the links for hotels
    elems = driver.find_elements_by_xpath("//a[@class = 'property_title prominent ']")
    for elem in elems:
        hotel_links.append(elem.get_attribute("href"))
    
    # find all the prices for different hotels
    #driver.get(url) # need to define it again!
    #prices = [_.text for _ in driver.find_elements_by_xpath("//div[@class = 'price __resizeWatch']")]
    
    """click 'next' button to next page"""
    # comment them off if you have only one page
    button_next = driver.find_element_by_xpath("//*[local-name() = 'a' and @class = 'nav next ui_button primary']")
    button_next.click()
    driver.execute_script("arguments[0].click();", button_next)
    url = driver.current_url
    

    
"""
Scraping from each link
"""
for l in range(len(hotel_links)):
    one_link = hotel_links[l]
    driver.get(one_link)
    time.sleep(randint(1,2)) # avoiding getting my IP address banned 
    
    # for each hotel page
    try:
        name = driver.find_element_by_xpath("//*[local-name()='h1'and @class='fkWsC b d Pn']").text
    except:
        name = ''
    
    try:
        star = driver.find_element_by_xpath("//span[@class='bvcwU P']").text
    except:
        star = ''
    
    try:
        rank = driver.find_element_by_xpath("//span[@class='dekGp Ci _R S4 H3 MD']").text
    except:
        rank = ''
    
    try:
        h_class = driver.find_element_by_xpath("//*[local-name() = 'svg' and @class = 'TkRkB d H0']").get_attribute("title")
    except:
        h_class = ''
    
    try:
        style = [_.text for _ in driver.find_elements_by_xpath("//div[@class='drcGn _R MC S4 _a H']")][1:]
    except:
        style = ''
    
    try:
        n_review = driver.find_element_by_xpath("//span[@class='HFUqL']").text
    except:
        n_review = ''
    
    try:
        grade_walker = driver.find_element_by_xpath("//*[local-name()='span'and @class='bpwqy dfNPK']").text
    except:
        grade_walker = ''
    
    try:
        n_restaurant = driver.find_element_by_xpath("//*[local-name()='span'and @class='bpwqy VyMdE']").text
    except:
        n_restaurant = ''
        
    try:
        n_attraction = driver.find_element_by_xpath("//*[local-name()='span'and @class='bpwqy eKwbS']").text 
    except:
        n_attraction = ''

    # get rating and reviews for 5 most recent answers
    try:
        ratings = [_.get_attribute("class") for _ in driver.find_elements_by_xpath("//div[@class='emWez F1']/span")]
    except:
        ratings = ''
    
    # click read-more to see full reviews
    try:
        button_readmore = driver.find_element_by_xpath("//span[@class='eljVo _S Z']")
        driver.execute_script("arguments[0].click();", button_readmore)
    except:
        pass  
    
    # now, scrape all the reviews possible    
    try:
        reviews = [_.text for _ in driver.find_elements_by_xpath("//q[@class='XllAv H4 _a']/span")]
    except:
        reviews = ''
    
    hotel_names.append(name)
    hotel_stars.append(star)
    hotel_ranks.append(rank)
    hotel_classes.append(h_class)
    hotel_styles.append(style)
    hotel_walkers.append(grade_walker)
    hotel_restaurant.append(n_restaurant)
    hotel_attraction.append(n_attraction)
    hotel_nreviews.append(n_review)
    # not all of them have more than 5 reviews
    try:
        hotel_ratings_0.append(ratings[0])
    except:
        hotel_ratings_0.append("")
    try:
        hotel_ratings_1.append(ratings[1])
    except:
        hotel_ratings_1.append("")
    try:
        hotel_ratings_2.append(ratings[2])
    except:
        hotel_ratings_2.append("")
    try:
        hotel_ratings_3.append(ratings[3])
    except:
        hotel_ratings_3.append("")
    try:
        hotel_ratings_4.append(ratings[4])
    except:
        hotel_ratings_4.append("")
    try:
        hotel_reviews_0.append(reviews[0])
    except:
        hotel_reviews_0.append("")
    try:
        hotel_reviews_1.append(reviews[1])
    except:
        hotel_reviews_1.append("")
    try:
        hotel_reviews_2.append(reviews[2])
    except:
        hotel_reviews_2.append("")
    try:
        hotel_reviews_3.append(reviews[3])
    except:
        hotel_reviews_3.append("")
    try:
        hotel_reviews_4.append(reviews[4])
    except:
        hotel_reviews_4.append("")
       
"""
close the driver
"""
driver.close()


"""
Store the result in dataframe
"""
df = pd.DataFrame(list(zip(hotel_names, hotel_stars, hotel_ranks, hotel_classes, hotel_styles, hotel_walkers, hotel_restaurant, hotel_attraction, hotel_nreviews, hotel_ratings_0, hotel_ratings_1, hotel_ratings_2, hotel_ratings_3, hotel_ratings_4, hotel_reviews_0, hotel_reviews_1, hotel_reviews_2, hotel_reviews_3, hotel_reviews_4)), columns = ['name', 'star', 'rank', 'class', 'style', 'grade_walkers', 'n_restaurants', 'n_attractions', 'n_reviews', 'ratings_0', 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'reviews_0', 'reviews_1', 'reviews_2', 'reviews_3', 'reviews_4'])
df.to_csv('web_scraping.csv', encoding='utf-8')








