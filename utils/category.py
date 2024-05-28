import os
import json
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv()
hugging_face_token = os.environ.get('HUGGING_FACE_TOKEN')

headers = {"Authorization": f"Bearer {hugging_face_token}"}

with open("Categories.json", "r") as file:
    category_object = json.load(file)

# Existing compare_sentences function with minor corrections


async def compare_sentences(session, source_sentence, sentences):
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
            async with session.post("https://api-inference.huggingface.co/models/sentence-transformers/msmarco-distilbert-base-tas-b", headers=headers, json=payload) as response:
                response.raise_for_status()
                response_json = await response.json()
                if 'error' in response_json:
                    print(f"Error: {response_json['error']}")
                    await asyncio.sleep(5)
                else:
                    return response_json

        except (aiohttp.ClientError, aiohttp.ClientResponseError, asyncio.TimeoutError) as e:
            print(f"An error occurred during API request: {e}")
            await asyncio.sleep(5)

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
        print("fourth_level_categories_dict==>", fourth_level_categories_dict, isinstance(fourth_level_categories_dict, list))
        if isinstance(fourth_level_categories_dict, list):
            fourth_level_categories = list(fourth_level_categories_dict)
        elif isinstance(fourth_level_categories_dict, dict):
            fourth_level_categories = list(fourth_level_categories_dict.keys())

        compare_result = await compare_sentences(session, keyword, fourth_level_categories)
        if compare_result is None:
            return {keyword: classification}

        max_value = max(compare_result)
        max_index = compare_result.index(max_value)
        fourth = fourth_level_categories[max_index]
        classification.append(fourth)

    except Exception as error:
        print(error)

    return {keyword: classification}


async def process_hashtags(hashtags):
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [classify(session, hashtag) for hashtag in hashtags]
        results = await asyncio.gather(*tasks)
    return results
