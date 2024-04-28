import re
from flask import Flask, render_template, request, make_response
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import BartTokenizer, BartForConditionalGeneration

app = Flask(__name__)

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def youtube_parser(url):
    video_id_regex = r'^(?:https?://)?(?:youtu\.be/|youtube\.com/(?:embed/|v/|watch\?v=))([\w-]{11})(?:\S+)?$'
    match = re.match(video_id_regex, url)
    return match.group(1) if match else None

def preprocess_text(text):
    # Remove music symbols
    text = re.sub(r'\[music\]', '', text)
    text = re.sub(r'\[.*?\]', '', text)

    # Other preprocessing steps as needed

    return text

def generate_summary(text, max_length=250, min_length=150):
    inputs = tokenizer.encode(text, return_tensors='pt', max_length=max_length, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, num_beams=2, early_stopping=True)
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/yts', methods=['GET', 'POST'])
def yts():
    if request.method == 'POST':
        v_ids = request.form['video-link']
        if not v_ids:
            message = 'Please enter a valid YouTube video URL or ID.'
            return render_template('yts.html', error=message)

        video_id = youtube_parser(v_ids)
        if video_id is not None:
            video_ids = [video_id]
        else:
            video_ids = re.findall(r'\b[\w-]{11}\b', v_ids)

        summary = ''
        for vid in video_ids:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(vid, languages=['en'])
                for item in transcript:
                    summary += item['text'] + ' '
            except:
                pass

        extracted_summary = preprocess_text(summary)
        generated_summary = generate_summary(extracted_summary, max_length=1024, min_length=300)

        return render_template('yts.html', summary=generated_summary)
    else:
        return render_template('yts.html')

@app.route('/download', methods=['POST'])
def download():
    if 'summary' in request.form:
        summary = request.form['summary']

        # Create a response with the summary text
        response = make_response(summary)

        # Set the appropriate headers for download
        response.headers.set('Content-Disposition', 'attachment', filename='summary.txt')
        response.headers.set('Content-Type', 'text/plain')

        return response
    else:
        return 'Invalid request'

if __name__ == '__main__':
    app.run(debug=True)
