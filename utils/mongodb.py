from datetime import datetime
from pymongo import MongoClient

mongo_client = MongoClient('mongodb+srv://krish:yXMdTPwSdTRo7qHY@serverlessinstance0.18otqeg.mongodb.net/')
db = mongo_client['Lurny']
collection = db['contents']

def save_content(url, content):
  """Function save content to database"""
  document = {
    'url': url,
    'content': content,
    'timestamp': datetime.utcnow()
  }
  collection.insert_one(document)
