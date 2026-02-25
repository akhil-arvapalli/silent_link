from flask import Flask, render_template, jsonify, request, send_from_directory
import speech_recognition as sr
import os
from threading import Thread
import string
import time

app = Flask(__name__)

# Global variables to maintain state
is_listening = False
current_transcript = ""
current_image = ""
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Test microphone access at startup
try:
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Microphone access successful!")
except Exception as e:
    print(f"Error accessing microphone: {e}")

# Complete ISL gif list
isl_gif = ['any questions', 'are you angry', 'are you busy', 'are you hungry', 'are you sick', 'be careful',
                'can we meet tomorrow', 'did you book tickets', 'did you finish homework', 'do you go to office', 'do you have money',
                'do you want something to drink', 'do you want tea or coffee', 'do you watch TV', 'dont worry', 'flower is beautiful',
                'good afternoon', 'good evening', 'good morning', 'good night', 'good question', 'had your lunch', 'happy journey',
                'hello what is your name', 'how many people are there in your family', 'i am a clerk', 'i am bore doing nothing',
                'i am fine', 'i am sorry', 'i am thinking', 'i am tired', 'i dont understand anything', 'i go to a theatre', 'i love to shop',
                'i had to say something but i forgot', 'i have headache', 'i like pink colour', 'i live in nagpur', 'lets go for lunch', 'my mother is a homemaker',
                'my name is john', 'nice to meet you', 'no smoking please', 'open the door', 'please call me later',
                'please clean the room', 'please give me your pen', 'please use dustbin dont throw garbage', 'please wait for sometime', 'shall I help you',
                'shall we go together tommorow', 'sign language interpreter', 'sit down', 'stand up', 'take care', 'there was traffic jam', 'wait I am thinking',
                'what are you doing', 'what is the problem', 'what is todays date', 'what is your father do', 'what is your job',
                'what is your mobile number', 'what is your name', 'whats up', 'when is your interview', 'when we will go', 'where do you stay',
                'where is the bathroom', 'where is the police station', 'you are wrong','address','agra','ahemdabad', 'all', 'april', 'assam', 'august', 'australia', 'badoda', 'banana', 'banaras', 'banglore',
                'bihar','bihar','bridge','cat', 'chandigarh', 'chennai', 'christmas', 'church', 'clinic', 'coconut', 'crocodile','dasara',
                'deaf', 'december', 'deer', 'delhi', 'dollar', 'duck', 'febuary', 'friday', 'fruits', 'glass', 'grapes', 'gujrat', 'hello',
                'hindu', 'hyderabad', 'india', 'january', 'jesus', 'job', 'july', 'july', 'karnataka', 'kerala', 'krishna', 'litre', 'mango',
                'may', 'mile', 'monday', 'mumbai', 'museum', 'muslim', 'nagpur', 'october', 'orange', 'pakistan', 'pass', 'police station',
                'post office', 'pune', 'punjab', 'rajasthan', 'ram', 'restaurant', 'saturday', 'september', 'shop', 'sleep', 'southafrica',
                'story', 'sunday', 'tamil nadu', 'temperature', 'temple', 'thursday', 'toilet', 'tomato', 'town', 'tuesday', 'usa', 'village',
                'voice', 'wednesday', 'weight','please wait for sometime','what is your mobile number','what are you doing','are you busy']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_listening():
    global is_listening
    is_listening = True
    Thread(target=listen_and_translate).start()
    return jsonify({"status": "success"})

@app.route('/stop', methods=['POST'])
def stop_listening():
    global is_listening
    is_listening = False
    return jsonify({"status": "success"})

@app.route('/status')
def get_status():
    return jsonify({
        "is_listening": is_listening,
        "transcript": current_transcript,
        "image": current_image
    })

@app.route('/ISL_Gifs/<path:filename>')
def serve_gif(filename):
    print(f"Attempting to serve gif: {filename}")  # Debug print
    try:
        return send_from_directory('ISL_Gifs', filename, mimetype='image/gif')
    except Exception as e:
        print(f"Error serving gif: {e}")  # Debug print
        return str(e), 404

@app.route('/letters/<path:filename>')
def serve_letter(filename):
    return send_from_directory('letters', filename)

def listen_and_translate():
    global current_transcript, current_image, is_listening

    with microphone as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening started!")

        while is_listening:
            try:
                print("Listening for speech...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                print("Audio captured, recognizing...")

                text = recognizer.recognize_google(audio).lower()
                print(f"Recognized: {text}")

                current_transcript = text

                # Remove punctuation
                for c in string.punctuation:
                    text = text.replace(c, "")

                if text in ['goodbye', 'good bye', 'bye']:
                    is_listening = False
                    break

                if text in isl_gif:
                    gif_path = f'{text}.gif'
                    print(f"Loading gif: {gif_path}")  # Debug print
                    if os.path.exists(os.path.join('ISL_Gifs', gif_path)):
                        current_image = f'/ISL_Gifs/{gif_path}'
                        print(f"Gif found, setting path: {current_image}")  # Debug print
                    else:
                        print(f"Gif not found: {gif_path}")  # Debug print
                else:
                    # Handle individual letters
                    for letter in text:
                        if letter in string.ascii_lowercase:
                            current_image = f'/letters/{letter}.jpg'
                            time.sleep(0.8)

            except sr.UnknownValueError:
                current_transcript = "Could not understand audio"
            except sr.RequestError:
                current_transcript = "Error with speech recognition service"
            except Exception as e:
                current_transcript = f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
