import requests

# Base URL for Wikipedia API
API_ENDPOINT = "https://en.wikipedia.org/w/api.php"

# Function to get section index by section title (heading)
def get_section_index(page_title, section_title):
    params = {
        'action': 'parse',
        'page': page_title,
        'prop': 'sections',
        'format': 'json'
    }
    response = requests.get(API_ENDPOINT, params=params)
    sections = response.json()['parse']['sections']
    
    for section in sections:
        if section['line'] == section_title:
            return section['index']
            
    return None

# Function to get content by section index
def get_section_content(page_title, section_index):
    params = {
        'action': 'parse',
        'page': page_title,
        'section': section_index,
        'prop': 'text',
        'format': 'json'
    }
    response = requests.get(API_ENDPOINT, params=params)
    return response.json()['parse']['text']['*']

# Example usage
page_title = 'Computer'  # Replace with actual page title
section_title = 'History'  # Replace with actual section title

section_index = get_section_index(page_title, section_title)
if section_index:
    section_content = get_section_content(page_title, section_index)
    print(section_content)  # This prints the HTML content of the section
else:
    print(f"Section '{section_title}' not found.")
