import io
import json
import logging
import os
import re
import time
from datetime import datetime 
import math
from logging.handlers import RotatingFileHandler

import PyPDF2
import requests
import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, jsonify, request, g
from flask_cors import CORS
# from gevent import monkey
from openai import OpenAI
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from youtube_transcript_api import YouTubeTranscriptApi

# monkey.patch_all()

ALLOWED_EXTENSIONS = {"pdf"}

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

app = Flask(__name__)
CORS(app)

mongo_client = MongoClient('mongodb+srv://nicolas1303563:awIKKGrgf4GmVYkV@cluster0.w3yzl84.mongodb.net/')
db = mongo_client['Lurny']
collection = db['contents']

# Configure the logger
app.logger.setLevel(logging.INFO)

# Create a RotatingFileHandler to log to a file
LOG_FILE = 'app.log'
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10000, backupCount=5)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
app.logger.addHandler(file_handler)

# Log a message to test the logging setup
app.logger.info('Application startup')

@app.before_request
def before_request():
    """Function before_request"""
    app.logger.info(f"Incoming request: {request.method} {request.url}")
    # g.openai_client = OpenAI(api_key=api_key)

@app.after_request
def after_request(response):
    """Function after_request"""    
    app.logger.info(f"Outgoing response: {response.status_code}")
    # if hasattr(g, 'openai_client'):
        # g.openai_client.close()
    return response

def get_transcript(video_id):
    """Function get_transcript from youtube video id"""        
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

def is_youtube_url(url):
    """Function check url is youtube url"""    
    
    return "youtube.com" in url or "youtu.be" in url


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Function get number of tokens"""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def split_content_evenly(content, parts):
    """Function split content as parts"""
    
    if parts <= 0:
        raise ValueError("Number of parts must be greater than zero.")
    
    part_len = len(content) // parts
    
    splits = []
    index = 0
    
    for _ in range(parts):
        splits.append(content[index: index + part_len])
        index += part_len
    return splits

def summarize(text):
    """Function summarize"""    
    
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
                            "items": {"type": "string"},
                            "description": "bulleted summary",
                        },
                        "hash_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "hash tag",
                        },
                    },
                    "required": ["title", "summary", "hash_tags"],
                },
            },
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """You have been provided with a passage. Read the provided passage and create an engaging title, like for example "How Does X Work?" or "7 Key Strategies to Achieve Y"). The title should not exceed 10 words. 

                Summary: Must Analyze the provided passage and extract 10 standalone learning briefs each of which should have at least 50 words. It is very important that there must be minimum 10 such learning briefs. It is very important that each learning briefs must have 50 words. This requirement is non-negotiable. Each learning brief must encapsulate a distinct aspect of the topic discussed, comprehensible in isolation. Focus on articulating key takeaways, trends, challenges, solutions, or impacts highlighted in the text. Ensure that each learning brief contributes to a deeper understanding of the subject matter without relying on context from other points. You should also make sure not to include acronyms without their expanded version because if these learning briefs were to be read in isolation, such acronyms would not make sense to the learner.

                Keywords: List all relevant keywords from the provided passage as hashtags. Additionally, include related keywords that might not be explicitly mentioned but are relevant to the subject matter. Include all proper nouns like names of people and places appearing in the provided passage as hashtags.
                """,
            },
            {"role": "user", "content": f"{text}"},
        ],
        tools=tools,
    )
    
    output = []
    for res in response.choices[0].message.tool_calls:
        output.append(res.function.arguments)

    return output[0]

def analyze(text):
    """Function analyze"""    

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
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a greate and sensitive assistant for generating multiple questions and answers"},
            {
                "role": "system",
                "content": "You have been provided with a passage. Split the provided passage in 5 equal parts. That is if the provided content has 1000 words, split the content into 5 parts of 200 words each. Then generate one multiple choice question for each of the 5 parts from the content. Each question should have 4 options, with one correct answer. The correct answer should be indicated separately. Additionally, provide a single explanation that will be displayed when a learner selects any of the wrong answers. The explanation should avoid direct references to the user's message.",
            },
            {"role": "user", "content": f"{text}"},
        ],
        tools=tools
    )    

    output = []
    for res in response.choices[0].message.tool_calls:
        output.append(res.function.arguments)
    return output

def extract_cover_image(url):
    """Function extract_cover_image"""      
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    cover_image = soup.find(
        "meta", property="og:image"
    ) 
    if cover_image:
        return cover_image["content"]
    cover_image = soup.find("meta", itemprop="image")  # Check for Schema.org image tag
    if cover_image:
        return cover_image["content"]
    cover_image = soup.find("img")  # Fallback to the first image tag on the page
    if cover_image:
        return cover_image["src"]
    return None

def save_content(url, content):
  """Function save content to database"""
  document = {
    'url': url,
    'content': content,
    'timestamp': datetime.utcnow()
  }
  collection.insert_one(document)

@app.route("/fetch", methods=["POST"])
def fetch_data_from_url():
    """Function fetch_data_from_url"""       
    try:
        data = request.get_json()
        url = data["url"]
        app.logger.info(f"Processing URL: {url}")

        # Initialize variables for the extracted text and media type
        combined_text = ""
        media = None
        cover_image_url = None

        start = time.time()

        # openai_client = g.openai_client
        # Extract text from the URL
        if is_youtube_url(url):
            # Extract video ID from the YouTube URL
            video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
            if video_id_match:
                video_id = video_id_match.group(1)
                try:
                    # Get the transcript from the YouTube video
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                    combined_text = "".join(entry["text"] for entry in transcript_list)
                    print("Video script", combined_text)
                except Exception as e:
                    return jsonify({"error": str(e)}), 400
            else:
                return jsonify({"error": "No YouTube video ID found in URL"}), 400

        elif url.lower().endswith(".pdf"):
            media = "PDF"
            try:
                response = requests.get(url)
                response.raise_for_status()  # Ensure we capture HTTP errors

                pdf_reader = PyPDF2.PdfReader(io.BytesIO(response.content))
                combined_text = "".join(
                    page.extract_text() + " " for page in pdf_reader.pages
                )

            except Exception as e:
                return jsonify({"error": str(e)}), 400
        else:
            media = "web"
            page = requests.get(url)

            soup = BeautifulSoup(page.content, "html.parser")

            cover_image_url = extract_cover_image(url)

            text_elements = [
                tag.get_text() for tag in soup.find_all(["p", "span", "a", "li"])
            ]
            combined_text = "".join(text_elements).strip()
        
        save_content(url, combined_text)
        
        num_tokens = num_tokens_from_string(combined_text, "gpt-3.5-turbo")
        
        num_parts = math.ceil(num_tokens / 10000) 
        print("num_parts", num_tokens, num_parts)
        splits = split_content_evenly(combined_text, num_parts) 
                
        results = []
        for split in splits:                        
            summary_content = summarize(split)
            question_content = analyze(split)

            json_string = json.dumps(
                {
                    "summary_content": summary_content,
                    "questions": question_content,
                    "image": cover_image_url if cover_image_url is not None else "",
                    "url": url,
                    "media": media,
                }
            )
            results.append(json_string)
        print(results)

        end = time.time()
        print(end - start, "s")

        return jsonify(results)

    except Exception as e:
        app.logger.error(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500
 
@app.route("/get_quiz", methods=["POST"])
def generage_quiz():
    """Function analyze"""    
    data = request.get_json()
    stub = data["stub"]

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
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a greate and sensitive assistant for generating multiple questions and answers"},
            {
                "role": "system",
                "content": "You have been provided with a 'learning brief' which encapsulates a distinct aspect of the topic discussed and capable of being comprehensible in isolation. In order for the learner to learn the key takeaway from the 'learning brief', you will create a multiple-choice question  to help memorize the given learning brief. Please provide four answer choices, with one being the correct choice that encapsulates the main idea of the statement. The correct answer should be indicated separately. Additionally, provide a single explanation that will be displayed when a learner selects any of the wrong answers.",
            },
            {"role": "user", "content": f"{stub}"},
        ],
        tools=tools
    )    

    output = []
    for res in response.choices[0].message.tool_calls:
        output.append(res.function.arguments)
        print("output", output)
    return output[0] 
   
@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    """Function upload_pdf"""    
    openai_client = g.openai_client
    # Check if the post request has the file part
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    # If user does not select file or submits an empty part without filename
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Secure the filename
        start = time.time()
        filename = secure_filename(file.filename)

        # Read the PDF content
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + " "

        # Do something with the extracted text
        # For example, returning it

        summary_content = summarize(text)
        question_content = analyze(text)

        print(summary_content, question_content)

        json_string = json.dumps(
            {
                "summary_content": summary_content,
                "questions": question_content,
                "fileName": filename,
                "media": "PDF",
            }
        )

        end = time.time()
        print(end - start, "s")

        return json_string

    return jsonify({"error": "Invalid file extension"}), 400

def allowed_file(filename):
    """Function allowed_file"""    
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5173,
        threaded=True,
        debug=True,
        ssl_context=(
            "/etc/letsencrypt/live/lurny.net/cert.pem",
            "/etc/letsencrypt/live/lurny.net/privkey.pem",
        ),
    )
