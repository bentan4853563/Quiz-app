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
from flask import Flask, jsonify, request, abort
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from youtube_transcript_api import YouTubeTranscriptApi
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.category import classify
from utils.mongodb import save_content
from utils.file_read import process_file
from utils.youtube import is_youtube_url
from utils.wikipedia import is_wikipedia_url, get_content_from_url, search_wikipedia, get_wikipedia_page_content
from utils.image import extract_cover_image
from utils.openai import summarize, quiz_from_stub, generate_quizes, from_content,  quiz
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
def lurnify_from_content():
    """Function fetch_data_from_url"""       
    try:
        data = request.get_json()
        combined_text = data["content"]

        result = from_content(combined_text)

        return result

    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
@app.route("/fetch", methods=["POST"])
def lurnify_from_url():
    """Function lurnify_from_url"""       
    try:
        data = request.get_json()
        url = data["url"]

        # Initialize variables for the extracted text and media type
        combined_text = ""
        media = None
        cover_image_url = None

        start = time.time()

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
        elif is_wikipedia_url(url):
            media = "wikipedia"
            try:
                combined_text = get_content_from_url(url)
            except Exception as e:
                return jsonify({"error": str(e)}), 400
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
                tag.get_text() for tag in soup.find_all(["p", "span", "a", "li","h1", "h2", "h3"])
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
            summary_content = summarize(split)
            print("after summarization")
            question_content = quiz(split)
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
def lurnify_from_file():
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
            summary_content = summarize(split)
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
def generage_quiz():
    """Function analyze"""    
    data = request.get_json()
    stub = data["stub"]
    print("stub", stub)
    quiz = quiz_from_stub(stub)

    return quiz

@app.route("/search", methods=["POST"])
def search_from_wiki():
    """Function that generates lurnies from the contents searched from Wikipedia"""
    print("search_from_wiki")
    
    # Try to parse JSON data or return a 400 error if the request is not JSON or is not properly formatted.
    try:
        data = request.get_json()
        if 'search_term' not in data:
            abort(400, description='search_term key is missing')
    except Exception as e:
        abort(400, description=str(e))

    search_term = data["search_term"]

    lurnies = []
    search_results = search_wikipedia(search_term)
    search_results = search_results[:2]
    if search_results:
        for article in search_results:
            page_content = get_wikipedia_page_content(article['pageid'])
            if page_content:
                lurny = from_content(str(page_content))
                lurnies.append(lurny)
    print(lurnies)
    return jsonify(lurnies)

@app.route("/collections-process", methods=["POST"])
def process_hashtags():
    data = request.get_json()
    print(data)
    hashtags = data["hashtag"]
    print("hashtag", hashtags)

    def classify_threaded(hashtags):
        return classify(hashtags)

    with ThreadPoolExecutor() as executor:
        # Submit all classification tasks to the thread pool
        future_to_hashtag = {executor.submit(classify_threaded, hashtag): hashtag for hashtag in hashtags}
        results = []

        # Iterate over the completed futures
        for future in as_completed(future_to_hashtag):
            hashtag = future_to_hashtag[future]
            try:
                result = future.result()
                print(result)
                results.append(result)
            except Exception as e:
                print(f"An error occurred during classification of hashtag {hashtag}: {e}")
                
    return results

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
        port=5115,
        threaded=True,
        debug=True,
        use_reloader=False,
        # ssl_context=(
        #     "/etc/letsencrypt/live/lurny.net/cert.pem",
        #     "/etc/letsencrypt/live/lurny.net/privkey.pem",
        # ),
    )
