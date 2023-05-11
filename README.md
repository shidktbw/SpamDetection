# Spam Detection for Gmail Messages
This Python script integrates Google Gmail API and a Machine Learning model for spam detection. The script reads the latest Gmail messages, predicts if they are spam or not, and outputs the results in the console.

# Usage

Before running this script, ensure that the 'spam_detection.h5' (the machine learning model file) and 'tokenizer.pickle' (the text tokenizer file) are in the same directory as the script.

  *  Go to your Google Cloud Console and create a new project.
  * Enable the Gmail API for that project.
  *  Download the JSON credentials file and rename it to 'credentials.json', then place it in the same directory as the script.
  *  Run the script. On the first run, it will open a new browser window asking you to authorize the script to access your Gmail account. Once authorized, it will create a 'token.pickle' file to remember the  authorization for future runs.
  
  The script will then fetch the latest 10 messages from your Gmail account, translate them into English if necessary, and use the machine learning model to predict if each message is spam or not.

The output will be printed in the console in the following format:
`Title: {subject of the message}
Message: This is probably spam. (or) This is probably not spam.`

If there is no text found in a message, it will print:
`Title: {subject of the message}
Message: Text didn't found`

# Script Details
The script uses the Gmail API to access the Gmail messages, BeautifulSoup to parse the email bodies, and the Translate library to translate non-English emails into English.

The spam detection is done using a TensorFlow model that was trained on a spam detection task. The model takes in the text of an email and outputs a number between 0 and 1, where a higher number indicates a higher likelihood that the email is spam. The threshold is set to 0.3, meaning any email with a score higher than this will be considered as spam.

# Disclaimer

Please note that the spam detection model might not be 100% accurate and there may be some false positives and negatives. Always double-check before deleting any emails. This script only reads emails and does not perform any actions like deleting or moving emails.
