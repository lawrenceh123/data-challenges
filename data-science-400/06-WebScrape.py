"""
Assignment 6 Web Scrape
Tally the number of links in an html page
"""

# 1. Import statements for necessary package(s)

import requests
from bs4 import BeautifulSoup 

###########################

# 2. Read in an html page from a freely and easily available source on the internet. 
# The html page must contain at least 3 links.

# read in following html page
url = 'https://www.crummy.com/self/resume.html'
# store as a request object
response = requests.get(url)
# get page content
content = response.content
# convert to BeautifulSoup object, use lxml’s HTML parser
soup = BeautifulSoup(content, 'lxml')
print(soup.prettify())
print('##########################################\n')

###########################

# 3. Write code to tally the number of links in the html page.

# use find_all to get info inside a tags, which commonly contains links
all_a = soup.find_all('a')
links = []
print('links within <a> tags:')
for i, x in enumerate(all_a):
    print(i+1, x.get('href'))
    links.append(x.get('href'))
print('##########################################\n')

# total number of links      
len(all_a)
# number of links with 'href' attribute (those without will appear as None)
len([x for x in links if x is not None])
# all links in this html page have 'href' attribtue 

###########################
      
# 4. Use print to present the tally

print('url:', url, '\nnumber of links:', len(all_a))

###########################
