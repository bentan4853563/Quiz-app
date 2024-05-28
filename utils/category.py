import os
import time
import json
import requests   # Use requests instead of aiohttp
from dotenv import load_dotenv

load_dotenv()
hugging_face_token = os.environ.get('HUGGING_FACE_TOKEN')

headers = {"Authorization": f"Bearer {hugging_face_token}"}

with open("Categories.json", "r") as file:
    category_object = json.load(file)

# Existing compare_sentences function with minor corrections


def compare_sentences(source_sentence, sentences):
    payload = {
        "inputs": {
            "source_sentence": source_sentence,
            "sentences": sentences
        }
    }

    max_retries = 5
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/sentence-transformers/msmarco-distilbert-base-tas-b",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            response_json = response.json()
            if 'error' in response_json:
                print(f"Error: {response_json['error']}")
                time.sleep(5)  # Blocking sleep since we are now using synchronous calls
            else:
                return response_json

        except requests.exceptions.RequestException as e:
            print(f"An error occurred during API request: {e}")
            time.sleep(5)  # Blocking sleep

        retry_count += 1

    print("Failed to get a valid response after retries.")
    return None

async def classify(session, keyword):
    try:
        classification = []

        # First level category is not classified here, moving directly to second level
        second_level_categories = [category for first_cat in category_object.values() for category in first_cat.keys()]
        
        compare_result = await compare_sentences(session, keyword, second_level_categories)
        if compare_result is None:
            return {keyword: classification}

        max_value = max(compare_result)
        max_index = compare_result.index(max_value)
        second = second_level_categories[max_index]
        first = [firstCat for firstCat, secondLevelDict in category_object.items() if second in secondLevelDict][0]
        
        classification.append(first)
        classification.append(second)

        third_level_categories = list(category_object[first][second].keys())
        compare_result = await compare_sentences(session, keyword, third_level_categories)
        if compare_result is None:
            return {keyword: classification}

        max_value = max(compare_result)
        max_index = compare_result.index(max_value)
        third = third_level_categories[max_index]
        classification.append(third)

        fourth_level_categories_dict = category_object[first][second][third]
        if fourth_level_categories_dict is None:
            return {keyword: classification}
        if isinstance(fourth_level_categories_dict, list):
            fourth_level_categories = list(fourth_level_categories_dict)
        elif isinstance(fourth_level_categories_dict, dict):
            fourth_level_categories = list(fourth_level_categories_dict.keys())

        compare_result = await compare_sentences( keyword, fourth_level_categories)
        if compare_result is None:
            return {keyword: classification}

        max_value = max(compare_result)
        max_index = compare_result.index(max_value)
        fourth = fourth_level_categories[max_index]
        classification.append(fourth)

    except Exception as error:
        print(error)

    return {keyword: classification}


def process_hashtags(hashtags):
    results = [classify(hashtag) for hashtag in hashtags]
    return results
