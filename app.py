import os
import re
import time
import json
import requests
import unicodedata
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from openai import OpenAI
from flask_cors import CORS

from youtube_transcript_api import YouTubeTranscriptApi

api_key = os.getenv('API_KEY')

key = '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'

youtube_api_key = "AIzaSyDBdkgyTOyrqEzdy_D4FQBiywDByzN5ukM"

client = OpenAI(api_key=key)

app = Flask(__name__)

CORS(app)

def get_transcript():
    video_id = 'iyabl4dlxbY'

    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    print(transcript)


def is_youtube_url(url):
    return 'youtube.com' in url or 'youtu.be' in url


def summarize(text):
     
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
                            "items": {
                                "type": "string"
                            },
                            "description": "bulleted summary"
                        },
                        "hash_tags": {
                            "type": "array", 
                            "items": {
                                "type": "string"
                            },
                            "description": "hash tag"
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
            {"role": "system", "content": f"Provide a concise title, not exceeding 10 words, that encapsulates the essence of the provided user's message. And create a summarized overview of the key learnings from the user's message. Ensure that the summary does not plagiarize the original text. Present the main points in 7 or more bulleted statements, focusing on the core themes and insights of the article. Avoid direct references to the article and instead, provide a coherent understanding of its subject matter. And list all the keywords found in the provided user's message as hashtags. Additionally, include relevant keyword hashtags that may not be explicitly mentioned in the article but are pertinent to the user's message."},
            {"role": "user", "content": text},
        ],
        tools=tools
    )

    output = []
    for res in response.choices[0].message.tool_calls:
        output.append(res.function.arguments)
    
    return output

def extract_hashtag(text):
    model = "gpt-4-turbo-preview"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"Your role is to extract keywords from the the user's message. Keywords shoud be a list of hashtag."},
            {"role": "user", "content": text},

        ]
    )

    output = []
    for res in response.choices[0].message.tool_calls:
        output.append(res.function.arguments)
    
    return output

def analyze(text):
    
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
                            "items": {
                                "type": "string"
                            },
                            "description": "available answer"
                        },
                        "correctanswer": {
                            "type": "string", 
                            "description": "correct answer"
                        },
                        "explanation": {
                            "type": "string", 
                            "description": "explanation"
                        },
                    },                    
                    "required": ["question", "answer", "correctanswer", "explanation"],
                },
            },
        }
    ]
    
    model = "gpt-4-turbo-preview"

    sample_result = """
    {"question": "What is the primary color of the sky during a clear day?","answer": ["Green", "Blue", "Red"],"correctanswer": "Blue","explanation": "The sky appears blue during a clear day due to the scattering of shorter wavelengths of light by molecules in the Earth's atmosphere.}
    """
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"Generate 5(not less, must 5) multiple choice questions based on the user's message . Each question should have 4 options, with one correct answer. The correct answer should be indicated separately. Additionally, provide a single explanation that will be displayed when a learner selects any of the wrong answers. The explanation should avoid direct references to the article."},
            {"role": "user", "content": text},
        ],
        tools = tools,
    )
    
    # result = response.choices[0].message.content
    
    # response = client.chat.completions.create(
    #     model=model,
    #     messages=[
    #         {"role": "system", "content": f"Your role is to systemize the user's message."},
    #         {"role": "user", "content": result},
    #     ],
    #     tools = tools,
    # )
    
    output = []
    for res in response.choices[0].message.tool_calls:
        output.append(res.function.arguments)
    
    return output

def extract_cover_image(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    cover_image = soup.find('meta', property='og:image')  # Check for Open Graph image tag
    if cover_image:
        return cover_image['content']
    cover_image = soup.find('meta', itemprop='image')  # Check for Schema.org image tag
    if cover_image:
        return cover_image['content']
    cover_image = soup.find('img')  # Fallback to the first image tag on the page
    if cover_image:
        return cover_image['src']
    return None

@app.route('/fetch', methods=['POST'])


def fetch_data_from_url():
    data = request.get_json()
    url = data['url']

    start = time.time()

    # Extract text from the URL
    if is_youtube_url(url):
        # Extract video ID from the YouTube URL
        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', url)
        if video_id_match:
            video_id = video_id_match.group(1)
            try:
                # Get the transcript from the YouTube video
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                video_transcript = ' '.join(entry['text'] for entry in transcript_list)
            except Exception as e:
                return jsonify({"error": str(e)}), 400
        else:
            return jsonify({"error": "No YouTube video ID found in URL"}), 400
    else:
        video_transcript = ""

    # Extract text from the URL
    page = requests.get(url)

    soup = BeautifulSoup(page.content, 'html.parser')

    cover_image_url = extract_cover_image(url)


    text_elements = [tag.get_text() for tag in soup.find_all(['p', 'span', 'a', 'li'])]
    extracted_text = ' '.join(text_elements).strip()

    # Combine the extracted text with the video transcript
    combined_text = extracted_text + " " + video_transcript

    print(combined_text)

    summary_content = summarize(combined_text)
    question_content = analyze(combined_text)

    print(summary_content)

    json_string = json.dumps({
        "summary_content": summary_content,
        "questions": question_content,
        "image": cover_image_url,
        "url": url
    })

    end = time.time()
    print(end - start, "s")

    return (json_string)

if __name__ == '__main__':
    app.run(debug=True, port=5173)
