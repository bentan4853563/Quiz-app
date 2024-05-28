import os
import json
import asyncio
import aiohttp

# Load environment variables and Categories.json
from dotenv import load_dotenv
load_dotenv()
hugging_face_token = os.environ.get('HUGGING_FACE_TOKEN')

headers = {"Authorization": f"Bearer {hugging_face_token}"}

with open("Categories.json", "r") as file:
    category_object = json.load(file)

async def compare_sentences(session, source_sentence, sentences):
    """Function compare_sentences"""
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
            async with aiohttp.ClientSession() as session:
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
    classification = []
    try:
        categories = []
        for first in list(category_object.keys()):
            for second in category_object[first].keys():
                categories.append(second)

        compare_result = await compare_sentences(session, keyword, categories)
        max_value = max(compare_result)
        max_index = compare_result.index(max_value)
        classification.append(list(categories)[max_index])
        for first in list(category_object.keys()):
            for second in category_object[first].keys():
                if second == categories[max_index]:
                    classification.insert(0, first)

        categories = list(category_object[classification[0]][classification[1]].keys())
        compare_result = await compare_sentences(session, keyword, categories)
        max_value = max(compare_result)
        max_index = compare_result.index(max_value)
        classification.append(list(categories)[max_index])

        print(type(category_object[classification[0]][classification[1]][classification[2]]))
        if type(category_object[classification[0]][classification[1]][classification[2]]) == "object":
            categories = list(category_object[classification[0]][classification[1]][classification[2]].keys())
        elif type(category_object[classification[0]][classification[1]][classification[2]]) == "dict":
            categories = list(category_object[classification[0]][classification[1]][classification[2]].values())

        compare_result = await compare_sentences(keyword, categories)
        max_value = max(compare_result)
        max_index = compare_result.index(max_value)
        classification.append(list(categories)[max_index])
        

    except Exception as error:
        print(f'Error classifying the keyword "{keyword}": {error}')
        return None
    return {keyword: classification}

# async def classify(session, keyword):
#     try:
#         classification = []
#         categories = []
#         category_paths = {}  # Dictionary to map categories to their paths

#         # Build the categories and store their paths
#         for firstCategory, secondLevel in category_object.items():
#             for secondCategory, thirdLevel in secondLevel.items():
#                 for thirdCategory, fourthLevel in thirdLevel.items():

#                     if isinstance(fourthLevel, list):
#                         for item in fourthLevel:
#                             categories.append(item)
#                             print("categories", categories)
#                             category_paths[item] = (firstCategory, secondCategory, thirdCategory)
#                     elif isinstance(fourthLevel, dict):
#                         for key in fourthLevel.keys():
#                             categories.append(key)
#                             category_paths[key] = (firstCategory, secondCategory, thirdCategory)
#         compare_result = await compare_sentences(session, keyword, categories)
        
#         # Find the max value and its corresponding index
#         max_value = max(compare_result)
#         max_index = compare_result.index(max_value)
        
#         # Get the category at max_index
#         max_category = categories[max_index]
        
#         # Retrieve the full category path using our category_paths dictionary
#         full_path = category_paths[max_category]

#         # Combine the path with the max value for the final classification
#         classification.extend(list(full_path))
#         classification.append(max_category)
#     except Exception as error:
#         print(error)

#     return {keyword: classification}

async def process_hashtags(hashtags):
    async with aiohttp.ClientSession(headers=headers) as session:
        # Using asyncio.gather to run classify tasks concurrently
        tasks = [classify(session, hashtag) for hashtag in hashtags]
        results = await asyncio.gather(*tasks)
    return results

