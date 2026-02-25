import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt
import cv2
from easygui import *
import os
from PIL import Image, ImageTk
from itertools import count
import tkinter as tk
import string
#import selecting
# obtain audio from the microphone
def func():
        r = sr.Recognizer()
        isl_gif=['any questions', 'are you angry', 'are you busy', 'are you hungry', 'are you sick', 'be careful',
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
        
        
        arr=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r', 's','t','u','v','w','x','y','z']
        with sr.Microphone() as source:
                # image   = "signlang.png"
                # msg="HEARING IMPAIRMENT ASSISTANT"
                # choices = ["Live Voice","All Done!"] 
                # reply   = buttonbox(msg,image=image,choices=choices)
                r.adjust_for_ambient_noise(source) 
                i=0
                while True:
                        print("I am Listening")
                        audio = r.listen(source)
                        # recognize speech using Sphinx
                        try:
                                a=r.recognize_google(audio)
                                a = a.lower()
                                print('You Said: ' + a.lower())
                                
                                for c in string.punctuation:
                                    a= a.replace(c,"")
                                    
                                if(a.lower()=='goodbye' or a.lower()=='good bye' or a.lower()=='bye'):
                                        print("oops!Time To say good bye")
                                        break
                                
                                elif(a.lower() in isl_gif):
                                    
                                    class ImageLabel(tk.Label):
                                            """a label that displays images, and plays them if they are gifs"""
                                            def load(self, im):
                                                if isinstance(im, str):
                                                    im = Image.open(im)
                                                self.loc = 0
                                                self.frames = []

                                                try:
                                                    for i in count(1):
                                                        self.frames.append(ImageTk.PhotoImage(im.copy()))
                                                        im.seek(i)
                                                except EOFError:
                                                    pass

                                                try:
                                                    self.delay = im.info['duration']
                                                except:
                                                    self.delay = 100

                                                if len(self.frames) == 1:
                                                    self.config(image=self.frames[0])
                                                else:
                                                    self.next_frame()

                                            def unload(self):
                                                self.config(image=None)
                                                self.frames = None

                                            def next_frame(self):
                                                if self.frames:
                                                    self.loc += 1
                                                    self.loc %= len(self.frames)
                                                    self.config(image=self.frames[self.loc])
                                                    self.after(self.delay, self.next_frame)
                                    root = tk.Tk()
                                    lbl = ImageLabel(root)
                                    lbl.pack()
                                    lbl.load(r'ISL_Gifs/{0}.gif'.format(a.lower()))
                                    root.mainloop()
                                else:
                                    for i in range(len(a)):
                                                    if(a[i] in arr):
                                            
                                                            ImageAddress = 'letters/'+a[i]+'.jpg'
                                                            ImageItself = Image.open(ImageAddress)
                                                            ImageNumpyFormat = np.asarray(ImageItself)
                                                            plt.imshow(ImageNumpyFormat)
                                                            plt.draw()
                                                            plt.pause(0.8)
                                                    else:
                                                            continue

                        except:
                               print(" ")
                        plt.close()

class SignLanguageTranslator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Indian Sign Language Translator")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.is_listening = False
        
        self.setup_gui()
        
    def setup_gui(self):
        # Create main container
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Style configuration
        style = ttk.Style()
        style.configure("Custom.TButton",
                       padding=10,
                       font=('Helvetica', 12))
        
        # Title
        title_label = ttk.Label(
            self.main_frame,
            text="Indian Sign Language Translator",
            font=('Helvetica', 24, 'bold')
        )
        title_label.pack(pady=20)
        
        # Status display
        self.status_label = ttk.Label(
            self.main_frame,
            text="Status: Ready",
            font=('Helvetica', 12)
        )
        self.status_label.pack(pady=10)
        
        # Display area for signs/gifs
        self.display_frame = ttk.Frame(
            self.main_frame,
            borderwidth=2,
            relief="solid"
        )
        self.display_frame.pack(pady=20, fill=tk.BOTH, expand=True)
        
        self.display_label = ttk.Label(self.display_frame)
        self.display_label.pack(pady=10)
        
        # Buttons frame
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(pady=20)
        
        # Start button
        self.start_button = ttk.Button(
            button_frame,
            text="Start Listening",
            style="Custom.TButton",
            command=self.toggle_listening
        )
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        # Exit button
        self.exit_button = ttk.Button(
            button_frame,
            text="Exit",
            style="Custom.TButton",
            command=self.root.quit
        )
        self.exit_button.pack(side=tk.LEFT, padx=10)
        
        # Transcript display
        self.transcript_label = ttk.Label(
            self.main_frame,
            text="Transcript: ",
            font=('Helvetica', 12)
        )
        self.transcript_label.pack(pady=10)

    def toggle_listening(self):
        if not self.is_listening:
            self.is_listening = True
            self.start_button.configure(text="Stop Listening")
            self.status_label.configure(text="Status: Listening...")
            self.listen_and_translate()
        else:
            self.is_listening = False
            self.start_button.configure(text="Start Listening")
            self.status_label.configure(text="Status: Stopped")

    def display_gif(self, gif_path):
        try:
            # Clear previous display
            self.display_label.configure(image='')
            
            # Load and display new gif
            gif = Image.open(gif_path)
            frames = []
            
            try:
                while True:
                    frames.append(ImageTk.PhotoImage(gif.copy()))
                    gif.seek(len(frames))
            except EOFError:
                pass
            
            def update_frame(frame_idx=0):
                if not self.is_listening:
                    return
                frame = frames[frame_idx]
                self.display_label.configure(image=frame)
                next_idx = (frame_idx + 1) % len(frames)
                self.root.after(100, update_frame, next_idx)
            
            update_frame()
            
        except Exception as e:
            print(f"Error displaying gif: {e}")

    def listen_and_translate(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            
            while self.is_listening:
                try:
                    self.status_label.configure(text="Status: Listening...")
                    audio = self.recognizer.listen(source)
                    text = self.recognizer.recognize_google(audio).lower()
                    
                    self.transcript_label.configure(text=f"Transcript: {text}")
                    
                    # Remove punctuation
                    for c in string.punctuation:
                        text = text.replace(c, "")
                    
                    if text in ['goodbye', 'good bye', 'bye']:
                        self.toggle_listening()
                        break
                    
                    if text in self.isl_gif:
                        gif_path = f'ISL_Gifs/{text}.gif'
                        self.display_gif(gif_path)
                    else:
                        # Display individual letters
                        for letter in text:
                            if letter in string.ascii_lowercase:
                                img_path = f'letters/{letter}.jpg'
                                if os.path.exists(img_path):
                                    img = Image.open(img_path)
                                    photo = ImageTk.PhotoImage(img)
                                    self.display_label.configure(image=photo)
                                    self.display_label.image = photo
                                    self.root.update()
                                    self.root.after(800)
                
                except sr.UnknownValueError:
                    self.status_label.configure(text="Status: Could not understand audio")
                except sr.RequestError:
                    self.status_label.configure(text="Status: Error with speech recognition service")
                except Exception as e:
                    self.status_label.configure(text=f"Status: Error - {str(e)}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = SignLanguageTranslator()
    app.run()
