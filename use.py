from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os.path
import base64
import re
from bs4 import BeautifulSoup
import pickle
from translate import Translator

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    try:
        service = build('gmail', 'v1', credentials=creds)
        return service
    except Exception as e:
        print(f'An error occurred: {e}')
        return None

def get_subject(payload):
    headers = payload['headers']
    for d in headers:
        if d['name'] == 'Subject':
            return d['value']
    return None



def parse_parts(parts):
    if 'body' in parts and 'data' in parts['body']:
        data = parts['body']['data']
        data = data.replace("-","+").replace("_","/")
        decoded_data = base64.b64decode(data)
        text = decoded_data.decode('utf-8')  
        return text
    return None  


def read_message(service, message):
    msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
    payload = msg['payload']
    data = {}
    if 'parts' in payload:
        parts = payload['parts']
        body = parse_parts(parts[0])
        if body is not None:
            translator= Translator(to_lang="en")
            translation = translator.translate(body) # translate to eng
            data['body'] = translation
        data['subject'] = get_subject(payload)
    return data



def predict_spam(message):
    sequence = tokenizer.texts_to_sequences([message])
    padded_sequence = pad_sequences(sequence, padding='post')
    prediction = model.predict(padded_sequence)
    return prediction[0]


model = load_model('spam_detection.h5')


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

service = get_service()
results = service.users().messages().list(userId='me').execute()
messages = results.get('messages', [])

latest_message = messages[0]
data = read_message(service, latest_message)

new_message = data['body']

messages = results.get('messages', [])[:10]  # only 10 new message <3

for message in messages:
    data = read_message(service, message)
    subject = data.get('subject', 'Teemat ei leitud')
    if 'body' in data:
        new_message = data['body']
        spam_prediction = predict_spam(new_message)
        if spam_prediction > 0.3:
            print(f"Title: {subject}\nMessage: This is probably spam.\n")
        else:
            print(f"Title: {subject}\nMessage: This is probably not spam.\n")
    else:
        print(f"Title: {subject}\nMessage: Text didnt found\n")





