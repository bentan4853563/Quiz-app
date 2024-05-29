import os
import time
import json
import requests   # Use requests instead of aiohttp
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed


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

# def classify(keyword):
#     try:
#         classification = []

#         # First level category is not classified here, moving directly to second level
#         second_level_categories = [category for first_cat in category_object.values() for category in first_cat.keys()]
        
#         compare_result = compare_sentences(keyword, second_level_categories)
#         if compare_result is None:
#             return {keyword: classification}

#         max_value = max(compare_result)
#         max_index = compare_result.index(max_value)
#         second = second_level_categories[max_index]
#         first = [firstCat for firstCat, secondLevelDict in category_object.items() if second in secondLevelDict][0]
        
#         classification.append(first)
#         classification.append(second)

#         third_level_categories = list(category_object[first][second].keys())
#         compare_result = compare_sentences(keyword, third_level_categories)
#         if compare_result is None:
#             return {keyword: classification}

#         max_value = max(compare_result)
#         max_index = compare_result.index(max_value)
#         third = third_level_categories[max_index]
#         classification.append(third)

#         fourth_level_categories_dict = category_object[first][second][third]
#         if fourth_level_categories_dict is None:
#             return {keyword: classification}
#         if isinstance(fourth_level_categories_dict, list):
#             fourth_level_categories = list(fourth_level_categories_dict)
#         elif isinstance(fourth_level_categories_dict, dict):
#             fourth_level_categories = list(fourth_level_categories_dict.keys())

#         compare_result = compare_sentences( keyword, fourth_level_categories)
#         if compare_result is None:
#             return {keyword: classification}

#         max_value = max(compare_result)
#         max_index = compare_result.index(max_value)
#         fourth = fourth_level_categories[max_index]
#         classification.append(fourth)

#     except Exception as error:
#         print(error)

#     return {keyword: classification}

# def classify(keyword):
#     try:
#         classification = []

#         # Flatten all category levels into a list of tuples containing the full path and the fourth level category
#         all_categories_with_paths = []
#         for first, second_level_dict in category_object.items():
#             for second, third_level_dict in second_level_dict.items():
#                 if isinstance(third_level_dict, dict):  # Ensure it's a dict before working with keys
#                     for third, fourth_level_list_or_dict in third_level_dict.items():
#                         if isinstance(fourth_level_list_or_dict, list):
#                             for fourth in fourth_level_list_or_dict:
#                                 all_categories_with_paths.append((first, second, third, fourth))
#                         elif isinstance(fourth_level_list_or_dict, dict):
#                             for fourth in fourth_level_list_or_dict.keys():
#                                 all_categories_with_paths.append((first, second, third, fourth))
#                         else:
#                             break
#                 else:
#                     break
#         # Extract just the fourth level categories for comparison
#         fourth_level_categories = [path[-1] for path in all_categories_with_paths]
#         print(len(fourth_level_categories))
#         # Find the best match in the fourth level categories
#         compare_results = compare_sentences(keyword, fourth_level_categories)
#         if compare_results is None or not compare_results[0]:
#             print("No matching fourth level category found.")
#             return {keyword: classification}
#         # Get the highest score index
#         max_value_index = compare_results.index(max(compare_results))
#         print(max_value_index)
#         best_match_path = all_categories_with_paths[max_value_index]
#         classification = best_match_path
#         print(keyword, ":", classification, "\n")
#     except Exception as error:
#         print(f"An error occurred during classification: {error}")

#     return {keyword: list(classification)}

def classify(keyword):
    try:
        classification = []

        # Flatten all category levels into a list of tuples containing the full path and the fourth level category
        all_categories_with_paths = []
        third_level_categories = []  # List to store third-level categories

        for first, second_level_dict in category_object.items():
            for second, third_level_dict in second_level_dict.items():
                if isinstance(third_level_dict, dict):  # Ensure it's a dict before working with keys
                    for third, fourth_level_list_or_dict in third_level_dict.items():
                        third_level_categories.append(third)  # Add third-level category to the list
                        if isinstance(fourth_level_list_or_dict, list):
                            for fourth in fourth_level_list_or_dict:
                                all_categories_with_paths.append((first, second, third, fourth))
                        elif isinstance(fourth_level_list_or_dict, dict):
                            for fourth in fourth_level_list_or_dict.keys():
                                all_categories_with_paths.append((first, second, third, fourth))
                        else:
                            break
                else:
                    break

        # Find the best match in the third-level categories
        compare_results = compare_sentences(keyword, third_level_categories)
        if compare_results is None or not compare_results[0]:
            print("No matching third level category found.")
            return {keyword: classification}

        # Get the highest score index for the third-level categories
        max_value_index = compare_results.index(max(compare_results))
        best_third_level_category = third_level_categories[max_value_index]

        # Find all fourth-level categories under the best third-level category
        corresponding_fourth_level_categories = [
            path for path in all_categories_with_paths if path[2] == best_third_level_category
        ]

        # Extract just the fourth-level categories for comparison
        fourth_level_categories = [path[-1] for path in corresponding_fourth_level_categories]
        if not fourth_level_categories:
            print("No corresponding fourth level categories found.")
            return {keyword: classification}

        # Find the best match in the fourth-level categories
        compare_results = compare_sentences(keyword, fourth_level_categories)
        if compare_results is None or not compare_results[0]:
            print("No matching fourth level category found.")
            return {keyword: classification}

        # Get the highest score index for the fourth-level categories
        max_value_index = compare_results.index(max(compare_results))
        best_match_path = corresponding_fourth_level_categories[max_value_index]
        classification = best_match_path
        print(keyword, ":", classification, "\n")

    except Exception as error:
        print(f"An error occurred during classification: {error}")

    return {keyword: list(classification)}

def process_hashtags(hashtags):
    def classify_threaded(hashtag):
        return classify(hashtag)

    with ThreadPoolExecutor() as executor:
        # Submit all classification tasks to the thread pool
        future_to_hashtag = {executor.submit(classify_threaded, hashtag): hashtag for hashtag in hashtags}
        results = []

        # Iterate over the completed futures
        for future in as_completed(future_to_hashtag):
            hashtag = future_to_hashtag[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"An error occurred during classification of hashtag {hashtag}: {e}")
                
    return results
