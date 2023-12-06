# -*- coding: utf-8 -*-
""" 
Web Scraping

Reviews are scraped from Trustpilot. The chosen company for the reviews is **BMO Financial Group**. The reviews are written to a CSV file with the following format:

| **companyName**| **datePublished** |**ratingValue**| **reviewBody** |
| ----------- | ----------- | ----------- | ----------- |
"""

import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import csv
import time
import warnings

# Ignore Warnings
warnings.filterwarnings("ignore")

# request the website first
resp = requests.get('https://ca.trustpilot.com/review/www.bmo.com')
print(resp)  # getting Response [200] is successful retrival

# save it so that it can be viewed in an editor
with open('resp.html', 'w') as f:
    f.write(resp.text)  # get the source code as a html file

# if request is successful, scrape all the text contents
if resp.status_code == 200:
    # read info
    soup = BeautifulSoup(resp.text)
    print(soup.title)
    # print(soup.text) # text in source code format


# Extract the total number of reviews.
item = soup.find(
    name='p',
    attrs={'class': 'typography_body-l__KUYFJ typography_appearance-default__AAY17'}
)
N = int(item.contents[0].replace(',', ''))
print('The total number of reviews for BMO is: ', N)


# Extract Company name
title = str(soup.title)

# search a substring in a raw string that starts with '<title>' and ends with 'Reviews'
pattern = r"<title>(.*?) Reviews"

search_result = re.search(pattern, title)
companyName = ''

# Check if a match is found
if search_result:
    # Extract the matched text (group 1 of the pattern)
    companyName = search_result.group(1)
    print("Company Name: ", companyName)
else:
    print("No company name found")

'''
Now, iterate over the review pages,
and from each page, extract the reviews, date when the review was published, the numerical value of the review rating.
Once the information are extracted, store the first 500 reviews to the CSV file in the format below:

    • comapnyName

    • datePublished, the

    • ratingValue, the

    • reviewBody, the review text.
'''

# Get a the reviewBody of a page


def get_page_review(soup):
    '''Scrape the review body from a provided webpage '''

    item = soup.find_all(
        name='div',
        attrs={'class': 'styles_reviewContent__0Q2Tg'}
    )
    pageReview = []
    for i in item:
        review = i.find(name='p', attrs={
                        'class': 'typography_body-l__KUYFJ typography_appearance-default__AAY17 typography_color-black__5LYEn'})
        # check if the reviewBody exists, if the review only has a header (no body), then assign empty string to the reviewBody of that review
        if review:
            pageReview.append(review.get_text())
        else:
            pageReview.append('')
    return pageReview


# Get the date of the reviews published of a page
def get_page_date(soup):
    '''Scrape the review published datetime from a provided webpage
    Note that this the review published datetime, not the date of the experience'''
    date = soup.find_all(
        name='div',
        attrs={'class': 'typography_body-m__xgxZ_ typography_appearance-subtle__8_H2l styles_datesWrapper__RCEKH'}

    )
    # replace the milliseconds and 'Z' with '+00:00'
    return [i.find('time')['datetime'].replace('.000Z', '+00:00') for i in date]


# Get the rating scores of the reviews of a page
def get_page_rating(soup):
    ''' Scrape the rating of the review from a provided webpage '''
    rating = soup.find_all(
        name='div',
        attrs={'class': 'styles_reviewHeader__iU9Px'}
    )
    return [int(i['data-service-review-rating']) for i in rating]


# Get the next page's url suffix
next_page = soup.find_all(name="a", string='Next page')[0].get('href')

# Initialize the data list with the first page information
datePublished = get_page_date(soup)
ratingValue = get_page_rating(soup)
reviewBody = get_page_review(soup)
page = 1

# Start to iterate through the pages of the review
while next_page:
    next_page_url = 'https://ca.trustpilot.com' + next_page
    # put 0.1 second interval between each scrape so we are not overloading the server
    time.sleep(.1)
    resp = requests.get(next_page_url)
    #print(resp, page)
    page += 1
    if resp.status_code == 200:
        # read info
        soup = BeautifulSoup(resp.text)
        # print("# of review", len(get_page_review(soup)))
        datePublished += get_page_date(soup)
        # print("# of date", len(get_page_date(soup)))
        ratingValue += get_page_rating(soup)
        reviewBody += get_page_review(soup)

        # move to next page
        next_page = soup.find_all(name="a", string='Next page')[0].get('href')
    else:
        print('Access Denied, Scraping Failed')
        break

print('Scraping Completed')

# Create a dictionary with column names as keys and lists as values
data = {
    'companyName': [companyName]*len(datePublished),
    'datePublished': datePublished,
    'ratingValue': ratingValue,
    'reviewBody': reviewBody
}

# Convert the dictionary into a Pandas DataFrame
df = pd.DataFrame(data)

# As suggested by Professor Hjalmar, skipping the reviews with empty reviewBody
# Note that hidden reviews are not included in this scraping
df.drop(df[df['reviewBody'] == ''].index, inplace=True)

# Output the dataframe (limit the number of reviews to the first 500) as a CSV file, by specifying 'encoding='utf-16'',
# we can keep the raw string of 'reviewBody' in the CSV file (as text like "it's" will be interpreted correctly)
df.iloc[:500].to_csv('bmo_review.csv', index=False, encoding='utf-16')
