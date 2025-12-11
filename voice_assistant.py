import json
import pyaudio
import time
import os
import sys
import re
import threading
import pyttsx3
from vosk import Model, KaldiRecognizer

class VoiceAssistant:
    def __init__(self, fetch_callback=None, log_callback=None):
        self.fetch_callback = fetch_callback
        self.log_callback = log_callback
        
        # Tools dictionary
        self.tools = {
            "screwdriver": ["screwdriver", "screw", "driver", "screwdriv", "skrewdriver"],
            "bolt": ["bolt", "bold", "board", "boat", "bult", "bolte", "bohlt"],
            "wrench": ["wrench", "rench", "range", "french", "spanner", "rensh"],
            "measuring tape": ["measuring tape", "measuring tap", "tape", "measure", "ruler", "measuring", "measur", "measure tape"],
            "hammer": ["hammer", "armor", "amor", "ammer", "hamer", "hamr", "mallet"],
            "plier": ["plier", "players", "pliers", "pryer", "player", "plyer", "pincher"]
        }
        
        # Initialize Text-to-Speech
        self.log("Initializing speech engine...")
        try:
            self.tts = pyttsx3.init()
            self.tts.setProperty('rate', 140)
            self.tts.setProperty('volume', 0.9)
        except Exception as e:
            self.log(f"Failed to initialize speech engine: {e}", "error")
            self.tts = None
        
        # Assistant settings
        self.robot_name = "Garage Assistant"
        self.is_speaking = False
        self.command_queue = []
        self.last_command_time = 0
        self.listening = False
        self.voice_thread = None
        
        # Load Vosk model
        model_path = "vosk-model-small-en-us-0.15"
        self.model = None
        if os.path.exists(model_path):
            try:
                self.model = Model(model_path)
                self.log("✅ Speech recognition ready!")
            except Exception as e:
                self.log(f"Failed to load Vosk model: {e}", "error")
        else:
            self.log(f"Vosk model not found at: {model_path}", "warning")
    
    def log(self, message, level="info"):
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(f"[{level.upper()}] {message}")
    
    def speak(self, text):
        """Make the assistant speak"""
        if not text or not self.tts:
            return
            
        self.log(f"🤖 Speaking: {text}")
        
        # Mark as speaking
        self.is_speaking = True
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except Exception as e:
            self.log(f"❌ Speaking error: {e}", "error")
            time.sleep(len(text.split()) * 0.3)  # Simulate speaking time
        
        # Mark as done speaking
        self.is_speaking = False
    
    def start_listening(self):
        """Start voice listening thread"""
        if not self.model:
            self.log("Cannot start listening - Vosk model not loaded", "error")
            return False
        
        if self.listening:
            self.log("Already listening", "info")
            return True
        
        self.listening = True
        self.voice_thread = threading.Thread(target=self.listen_continuously, daemon=True)
        self.voice_thread.start()
        self.log("🎤 Voice assistant started - Always listening...", "success")
        self.speak("Voice control activated. I'm listening for your commands.")
        return True
    
    def stop_listening(self):
        """Stop voice listening"""
        self.listening = False
        if self.voice_thread:
            self.voice_thread.join(timeout=1.0)
        self.log("Voice assistant stopped", "info")
    
    def listen_continuously(self):
        """Always listen in background"""
        try:
            p = pyaudio.PyAudio()
            
            # Find available microphone
            device_index = None
            info = p.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            
            for i in range(0, numdevices):
                if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                    device_index = i
                    break
            
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8000,
                input_device_index=device_index
            )
            
            recognizer = KaldiRecognizer(self.model, 16000)
            
            self.log("🔊 Always listening for voice commands...", "info")
            
            while self.listening:
                try:
                    # Don't process audio while speaking
                    if self.is_speaking:
                        time.sleep(0.1)
                        continue
                    
                    data = stream.read(4000, exception_on_overflow=False)
                    
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text = result.get('text', '').strip()
                        if text and len(text) > 2:
                            current_time = time.time()
                            # Debounce: ignore if too soon after last command
                            if current_time - self.last_command_time > 1.5:
                                self.log(f"🎤 Heard: {text}", "info")
                                self.command_queue.append(text)
                                self.last_command_time = current_time
                    
                except Exception as e:
                    if self.listening:  # Only log if we're supposed to be listening
                        self.log(f"Listening error: {e}", "warning")
                    time.sleep(0.1)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            self.log(f"Failed to start audio stream: {e}", "error")
    
    def process_voice_commands(self):
        """Process queued voice commands"""
        while self.command_queue:
            text = self.command_queue.pop(0)
            self.process_command(text)
    
    def find_tool(self, text):
        """Find which tool was mentioned"""
        text_lower = text.lower()
        
        # Check each tool's keywords
        for tool, keywords in self.tools.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return tool
        
        # Check for similar words (first 3 letters)
        words = text_lower.split()
        for word in words:
            if len(word) >= 3:
                for tool, keywords in self.tools.items():
                    for keyword in keywords:
                        if word[:3] == keyword[:3]:
                            return tool
        
        return None
    
    def process_command(self, text):
        """Process a user command"""
        text_lower = text.lower()
        
        # Check for help
        if any(word in text_lower for word in ["help", "what can you do", "instructions"]):
            self.speak("I can fetch tools for you. Just say the name of the tool, like 'hammer' or 'wrench'.")
            return True, None

        # Check for tool request
        tool = self.find_tool(text_lower)
        
        if tool:
            # Announce fetching
            fetching_msg = f"Fetching the {tool} for you!"
            self.speak(fetching_msg)
            
            # Return the tool name to be fetched
            return True, tool

        # Check for greeting
        if any(greet in text_lower for greet in ["hello", "hi", "hey", "greetings"]):
            self.speak("Hello! I'm your Garage Assistant. Say the name of a tool you need.")
            return True, None

        # Check for stop/quit
        if any(word in text_lower for word in ["stop", "quit", "exit", "shut down"]):
            self.speak("Goodbye! Voice control will stop.")
            return False, None

        # If nothing understood
        self.speak("I didn't catch that. Please say the name of a tool, like 'hammer' or 'screwdriver'.")
        return True, None
    
    def get_fetching_message(self, tool):
        """Generate fetching message"""
        return f"Fetching the {tool} for you!"
    
    def get_confirmation_message(self, tool):
        """Generate confirmation message after fetching"""
        return f"Here is your {tool}! What else can I get for you?"