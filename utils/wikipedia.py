import re
import requests

def is_wikipedia_url(url):
    # Regular expression pattern for a Wikipedia URL
    wikipedia_pattern = r'^https?://([a-z]{2}\.)?wikipedia\.org/wiki/[^ ]+$'
    
    # Match the URL against the pattern
    match = re.match(wikipedia_pattern, url)
    
    return bool(match)# Example usage:

def search_wikipedia(search_term):
    # Define the endpoint and parameters
    url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': search_term,
        'format': 'json'
    }

    # Make the request to Wikipedia's API
    response = requests.get(url, params=params)
    # Check for a successful request
    if response.status_code == 200:
        return response.json()['query']['search']
    else:
        # In case of an error, print the status code and return None
        print(f"Error: {response.status_code}")
        return None

def get_content_from_url(url):
    # Extract the title from the URL
    title = url.split('/')[-1]

    # Define the endpoint and parameters for getting page content by title
    api_url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'titles': title,
        'prop': 'extracts|info',
        'inprop': 'url',
        'explaintext': True,
        'format': 'json'
    }

    # Make the request to Wikipedia's API
    response = requests.get(api_url, params=params)

    # Check for a successful request
    if response.status_code == 200:
        pages = response.json()['query']['pages']
        pageid = next(iter(pages))
        if pageid != "-1":
            page = pages[pageid]
            extract = page['extract']
            return extract
        else:
            print("Page not found.")
            return None
    else:
        # In case of an error, print the status code and return None
        print(f"Error: {response.status_code}")
        return None

def get_wikipedia_page_content(pageid):
    # Define the endpoint and parameters for getting page content by pageid
    url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'pageids': pageid,
        'prop': 'extracts|info',
        'inprop': 'url',
        'explaintext': True,
        'format': 'json'
    }

    # Make the request to Wikipedia's API
    response = requests.get(url, params=params)

    # Check for a successful request
    if response.status_code == 200:
        pages = response.json()['query']['pages']
        if str(pageid) in pages:
            page = pages[str(pageid)]
            title = page['title']
            extract = page['extract']
            fullurl = page['fullurl']
            return {
                'title': title,
                'content': extract,
                'url': fullurl
            }
        else:
            print("Page not found")
            return None
    else:
        # In case of an error, print the status code and return None
        print(f"Error: {response.status_code}")
        return None

# # Example usage with previous search function
# search_results = search_wikipedia('Python programming')
# if search_results:
#     for article in search_results:
#         page_content = get_wikipedia_page_content(article['pageid'])
#         if page_content:
#             print(f"Title: {page_content['title']}\nURL: {page_content['url']}\nContent: {page_content['content']}\n")
