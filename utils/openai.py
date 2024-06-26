import os
import json
import time
import math
from openai import OpenAI
from flask import jsonify
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from utils.mongodb import save_content
from utils.content import split_content_evenly
from utils.content import num_tokens_from_string

load_dotenv()

MAX_RETRIES = 5  
RETRY_DELAY = 1

MODEL = "gpt-3.5-turbo"
UPLOAD_FOLDER = "uploads"
SUMMARY_PROMPT_FILE_PATH = 'Prompts/summary.txt'
QUIZ_PROMPT_FILE_PATH = 'Prompts/quiz.txt'
STUB_PROMPT_FILE_PATH = 'Prompts/stub.txt'

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def summarize(text):
    """Function summarize with retry logic"""    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_title_summary_hashtags",
                "description": "Get title, summary and hash_tags from use's message.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "title",
                        },
                        "summary": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "question"
                                    },
                                    "answer": {
                                        "type": "string",
                                        "description": "answer"
                                    }
                                },
                                "required": ["question", "answer"]
                            }
                            
                        },
                        "hash_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "hash tags",
                        },
                    },
                    "required": ["title", "summary", "hash_tags"],
                },
            },
        }
    ]

    attempts = 0

    while attempts < MAX_RETRIES:
        try:
            with open(SUMMARY_PROMPT_FILE_PATH, encoding='utf-8') as file:
                prompt = file.read()
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ],
                tools=tools,
            )

            # Check if the response contains the expected output
            if response.choices[0].message.tool_calls is not None:
                output = []
                for res in response.choices[0].message.tool_calls:
                    output.append(res.function.arguments)
                result = json.loads(output[0])
                # print("Before processing HashTag")
                # collections = process_hashtags(result["hash_tags"])
                # result["hash_tags"] = collections
                return result
            
            # If we got a response but tool_calls is None, raise an exception to retry
            else:
                raise ValueError("Received None from tool_calls")

        except (ValueError, AttributeError) as e:
            attempts += 1
            print(f"Attempt {attempts} failed (Summarize) with error: {e}. Retrying...")
            time.sleep(RETRY_DELAY)

    # If we've exceeded the maximum number of retries, raise an exception
    raise RuntimeError("Max retries exceeded. Unable to get a valid response.")

def generate_quizes(text):
    """Function Generate Quizzes"""
    # Assuming split_content_evenly is a function that splits the text into even parts
    splits = split_content_evenly(text, 5)
    
    quizzes = []
    # Use ThreadPoolExecutor to execute tasks asynchronously
    with ThreadPoolExecutor() as executor:
        # Map the generate_quiz function over the splits
        future_quizzes = executor.map(quiz, splits)
        
        for future_quiz in future_quizzes:
            quizzes.append(future_quiz)
    
    return quizzes   

def quiz(text):
    """Function Generate Quiz"""

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_question_answer",
                "description": "Generate question and answers, correctanswer, explanation for the question",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "question",
                        },
                        "answer": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "answers",
                        },
                        "correctanswer": {
                            "type": "string",
                            "description": "correct answer",
                        },
                        "explanation": {"type": "string", "description": "explanation"},
                    },
                    "required": ["question", "answer", "correctanswer", "explanation"],
                },
            },
        }
    ]
    attempts = 0

    while attempts < MAX_RETRIES:
        try:
            with open(QUIZ_PROMPT_FILE_PATH, encoding='utf-8') as file:
                prompt = file.read()

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ],
                tools=tools,
            )
            print("token number of split", num_tokens_from_string(text, MODEL))

            if not response.choices:  # Check if choices array is empty
                raise ValueError("No choices available in response")

            if response.choices[0].message.tool_calls is not None:
                output = []
                for res in response.choices[0].message.tool_calls:
                    output.append(res.function.arguments)
                return output[0]

            # If we got a response but tool_calls is None, raise an exception to retry
            else:
                raise ValueError("Received None from tool_calls")

        except (ValueError, AttributeError) as e:
            attempts += 1
            print(f"Attempt {attempts} failed (Quiz) with error: {e}. Retrying...")
            time.sleep(RETRY_DELAY)

    # If we've exceeded the maximum number of retries, raise an exception
    raise RuntimeError("Max retries exceeded. Unable to get a valid response.")

def quiz_from_stub(stub):
    """Function generate quiz from stub"""   

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_question_answer",
                "description": "Generate question and answers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "question",
                        },
                        "answer": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "available answer",
                        },
                        "correctanswer": {
                            "type": "string",
                            "description": "correct answer",
                        },
                        "explanation": {"type": "string", "description": "explanation"},
                    },
                    "required": ["question", "answer", "correctanswer", "explanation"],
                },
            },
        }
    ]    
    
    with open(STUB_PROMPT_FILE_PATH, encoding='utf-8') as file:
        prompt = file.read()
        
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": stub},
        ],
        tools=tools,
    )    

    output = []
    for res in response.choices[0].message.tool_calls:
        output.append(res.function.arguments)
    return output[0]

def from_content(content):
    media = None
    url = None
    cover_image_url = None

    start = time.time()

    # Save content to DB
    save_content(None, content)
    
    # Calulate number of tokens for the certain gpt model
    num_tokens = num_tokens_from_string(content, MODEL)
    print(num_tokens, len(content))
    
    num_parts = math.ceil(num_tokens / 15000) 
    # Split entire content to several same parts when content is large than gpt token limit
    splits = split_content_evenly(content, num_parts) 
            
    results = []
    for split in splits: 
        print("before summarization")
        summary_content = summarize(split)
        print("after summarization")
        question_content = generate_quizes(split)
        print("after quiz")

        result = {
            "summary_content": summary_content,
            "questions": question_content,
            "image": cover_image_url if cover_image_url is not None else "",
            "url": url,
            "media": media,
        }
        results.append(result)

    end = time.time()
    print(end - start, "s")

    return results