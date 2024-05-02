"""Module providing a system that extracting data from PDF and process by OpenAI"""

import io
import os
import re
import time
import json
import logging
from logging.handlers import RotatingFileHandler
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, g
from werkzeug.utils import secure_filename
import PyPDF2
from openai import OpenAI
from flask_cors import CORS
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi


ALLOWED_EXTENSIONS = {"pdf"}

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# client = OpenAI(api_key=api_key)

app = Flask(__name__)

CORS(app)

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
    g.openai_client = OpenAI(api_key=api_key)

@app.after_request
def after_request(response):
    """Function after_request"""    
    app.logger.info(f"Outgoing response: {response.status_code}")
    if hasattr(g, 'openai_client'):
        g.openai_client.close()
    return response

def get_transcript(video_id):
    """Function get_transcript"""    
    
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    print(transcript)


def is_youtube_url(url):
    """Function is_youtube_url"""    
    
    return "youtube.com" in url or "youtu.be" in url


def extract_hashtag(text, client):
    """Function extract_hashtag"""    
    
    model = "gpt-4-turbo-preview"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Your role is to extract keywords from the the user's message. Keywords shoud be a list of hashtag.",
            },
            {"role": "user", "content": text},
        ],
    )

    output = []
    for res in response.choices[0].message.tool_calls:
        output.append(res.function.arguments)

    return output[0]


def summarize(text, client):
    """Function summarize"""    

    model = "gpt-4-turbo-preview"

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
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Provide a concise title, not exceeding 10 words, that encapsulates the essence of the provided user's message. And create a summarized overview of the key learnings from the user's message. Ensure that the summary does not plagiarize the original text. Present the main points in 7 or more bulleted statements, focusing on the core themes and insights of the article. Avoid direct references to the article and instead, provide a coherent understanding of its subject matter. And list all the keywords found in the provided user's message as hashtags. Additionally, include relevant keyword hashtags that may not be explicitly mentioned in the article but are pertinent to the user's message.",
            },
            {"role": "user", "content": text},
        ],
        tools=tools,
    )
    output = []
    for res in response.choices[0].message.tool_calls:
        output.append(res.function.arguments)

    return output[0]


def analyze(text, client):
    """Function analyze"""    

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_question_answer",
                "description": "Get questions and answers from use's message.",
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

    model = "gpt-4-turbo-preview"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Generate 5(not less, must 5) multiple choice questions based on the user's message . Each question should have 4 options, with one correct answer. The correct answer should be indicated separately. Additionally, provide a single explanation that will be displayed when a learner selects any of the wrong answers. The explanation should avoid direct references to the article.",
            },
            {"role": "user", "content": text},
        ],
        tools=tools,
    )

    output = []
    for res in response.choices[0].message.tool_calls:
        output.append(res.function.arguments)

    return output[0]


def extract_cover_image(url):
    """Function extract_cover_image"""    
    
    page = requests.get(url, timeout=10)
    soup = BeautifulSoup(page.content, "html.parser")
    cover_image = soup.find(
        "meta", property="og:image"
    )  # Check for Open Graph image tag
    if cover_image:
        return cover_image["content"]
    cover_image = soup.find("meta", itemprop="image")  # Check for Schema.org image tag
    if cover_image:
        return cover_image["content"]
    cover_image = soup.find("img")  # Fallback to the first image tag on the page
    if cover_image:
        return cover_image["src"]
    return None


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

        openai_client = g.openai_client
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
                response = requests.get(url, timeout=10)
                response.raise_for_status()  # Ensure we capture HTTP errors

                pdf_reader = PyPDF2.PdfReader(io.BytesIO(response.content))
                combined_text = "".join(
                    page.extract_text() + " " for page in pdf_reader.pages
                )

            except Exception as e:
                return jsonify({"error": str(e)}), 400
        else:
            media = "web"
            page = requests.get(url, timeout=10)

            soup = BeautifulSoup(page.content, "html.parser")

            cover_image_url = extract_cover_image(url)

            text_elements = [
                tag.get_text() for tag in soup.find_all(["p", "span", "a", "li"])
            ]
            combined_text = "".join(text_elements).strip()

        print(combined_text)

        summary_content = summarize(combined_text, openai_client)
        question_content = analyze(combined_text, openai_client)

        json_string = json.dumps(
            {
                "summary_content": summary_content,
                "questions": question_content,
                "image": cover_image_url if cover_image_url is not None else "",
                "url": url,
                "media": media,
            }
        )

        print(json_string + "\n")

        end = time.time()
        print(end - start, "s")

        return json_string

    except Exception as e:
        app.logger.error(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500
    
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
        print(text)

        summary_content = summarize(text, openai_client)
        question_content = analyze(text, openai_client)

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
        ssl_context=(
            "/etc/letsencrypt/live/lurny.net/cert.pem",
            "/etc/letsencrypt/live/lurny.net/privkey.pem",
        ),
    )
