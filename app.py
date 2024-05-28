import io
import json
import os
import re
import time
import math
import PyPDF2
import requests
# from gevent import monkey
from flask_cors import CORS
from flask import Flask, jsonify, request
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from youtube_transcript_api import YouTubeTranscriptApi

from utils.mongodb import save_content
from utils.file_read import process_file
from utils.youtube import is_youtube_url
from utils.image import extract_cover_image
from utils.openai import summarize, quiz_from_stub, generate_quizes
from utils.content import num_tokens_from_string, split_content_evenly
from utils.prompt import read_prompt_file, update_prompt_file

load_dotenv()

app = Flask(__name__)
CORS(app)

MODEL = "gpt-3.5-turbo"

SUMMARY_PROMPT_FILE_PATH = 'Prompts/summary.txt'
QUIZ_PROMPT_FILE_PATH = 'Prompts/quiz.txt'
STUB_PROMPT_FILE_PATH = 'Prompts/stub.txt'

@app.route("/manually", methods=["POST"])
async def lurnify_from_content():
    """Function fetch_data_from_url"""       
    try:
        data = request.get_json()
        combined_text = data["content"]

        media = None
        url = None
        cover_image_url = None

        start = time.time()

        # Save content to DB
        save_content(None, combined_text)
        
        # Calulate number of tokens for the certain gpt model
        num_tokens = num_tokens_from_string(combined_text, MODEL)
        print(num_tokens, len(combined_text))
        
        num_parts = math.ceil(num_tokens / 15000) 
        # Split entire content to several same parts when content is large than gpt token limit
        splits = split_content_evenly(combined_text, num_parts) 
                
        results = []
        for split in splits: 
            print("before summarization")
            summary_content = await summarize(split)
            print("after summarization")
            question_content = generate_quizes(split)
            print("after quiz")

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

        end = time.time()
        print(end - start, "s")

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
@app.route("/fetch", methods=["POST"])
async def lurnify_from_url():
    """Function lurnify_from_url"""       
    try:
        data = request.get_json()
        url = data["url"]

        # Initialize variables for the extracted text and media type
        combined_text = ""
        media = None
        cover_image_url = None

        start = time.time()
        print("=>url", url, is_youtube_url(url))

        # Extract text from the URL
        if is_youtube_url(url):
            # Extract video ID from the YouTube URL
            video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
            if video_id_match:
                video_id = video_id_match.group(1)
                try:
                    # Get the transcript from the YouTube video
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])
                    combined_text = "".join(entry["text"] for entry in transcript_list)
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
            combined_text = ", ".join(text_elements)
        
        # Save content to DB
        save_content(url, combined_text)
        
        # Calulate number of tokens for the certain gpt model
        num_tokens = num_tokens_from_string(combined_text, MODEL)
        print(num_tokens, len(combined_text))
        
        num_parts = math.ceil(num_tokens / 15000) 
        # Split entire content to several same parts when content is large than gpt token limit
        splits = split_content_evenly(combined_text, num_parts) 
                
        results = []
        for split in splits:                        
            print("before summarization")
            summary_content = await summarize(split)
            print("after summarization")
            question_content = generate_quizes(split)
            print("after quiz")

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
        end = time.time()
        print(end - start, "s")

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/lurnify-from-file', methods=['POST'])
async def lurnify_from_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join("uploads", filename)
        file.save(file_path)
        text = process_file(file_path, filename.rsplit('.', 1)[1].lower())
        start = time.time()

        # Save content to DB
        # save_content(None, text)
        
        # Calulate number of tokens for the certain gpt model
        num_tokens = num_tokens_from_string(text, MODEL)
        print(num_tokens, len(text))
        
        num_parts = math.ceil(num_tokens / 15000) 
        # Split entire content to several same parts when content is large than gpt token limit
        splits = split_content_evenly(text, num_parts) 
                
        results = []
        for split in splits:                        
            summary_content = await summarize(split)
            question_content = generate_quizes(split)

            json_string = json.dumps(
                {
                    "summary_content": summary_content,
                    "questions": question_content,
                    "image": None,
                    "url": filename,
                    "media": "file",
                }
            )
            results.append(json_string)

        end = time.time()
        print(end - start, "s")

        return jsonify(results)
 
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route("/get_quiz", methods=["POST"])
async def generage_quiz():
    """Function analyze"""    
    data = request.get_json()
    stub = data["stub"]
    quiz = quiz_from_stub(stub)

    return quiz

@app.route("/update_prompts", methods=["POST"])
def update_prompts():
    """Endpoint to update prompt files based on JSON input"""
    data = request.get_json()
    
    summary_prompt = data.get('summary')
    quiz_prompt = data.get('quiz')
    stub_prompt = data.get('stub')
    
    # Check if any of the prompts are provided and update accordingly
    if summary_prompt and not update_prompt_file(SUMMARY_PROMPT_FILE_PATH, summary_prompt):
        return jsonify({"error": f"Failed to update summary prompt"}), 500
    
    if quiz_prompt and not update_prompt_file(QUIZ_PROMPT_FILE_PATH, quiz_prompt):
        return jsonify({"error": "Failed to update quiz prompt"}), 500
    
    if stub_prompt and not update_prompt_file(STUB_PROMPT_FILE_PATH, stub_prompt):
        return jsonify({"error": "Failed to update stub prompt"}), 500
    
    return jsonify({"message": "Prompts updated successfully"}), 200

@app.route("/get_prompts", methods=["GET"])
def get_prompts():
    """Endpoint to get the current prompts"""
    summary_prompt = read_prompt_file(SUMMARY_PROMPT_FILE_PATH)
    quiz_prompt = read_prompt_file(QUIZ_PROMPT_FILE_PATH)
    stub_prompt = read_prompt_file(STUB_PROMPT_FILE_PATH)

    # Ensure all prompts were read successfully
    if summary_prompt is None or quiz_prompt is None or stub_prompt is None:
        return jsonify({"error": "Failed to read one or more prompt files"}), 500
    
    # Return the prompts in JSON format
    prompts = {
        "summary": summary_prompt,
        "quiz": quiz_prompt,
        "stub": stub_prompt,
    }
    
    return jsonify(prompts), 200

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'pptx', 'txt'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5173,
        threaded=True,
        debug=True,
        # use_reloader=False,
        # ssl_context=(
        #     "/etc/letsencrypt/live/lurny.net/cert.pem",
        #     "/etc/letsencrypt/live/lurny.net/privkey.pem",
        # ),
    )
