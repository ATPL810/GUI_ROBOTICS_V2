import os
import time
import threading
import json
import re
import pyaudio
import pyttsx3
from vosk import Model, KaldiRecognizer

class VoiceRecognitionSystem:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.running = False
        self.voice_enabled = False
        self.is_speaking = False
        self.command_queue = []
        self.last_command_time = 0
        self.command_callback = None
        self.paused = False
        
        # Tools dictionary
        self.tools = {
            "screwdriver": ["screwdriver", "screw", "driver", "screwdriv", "skrewdriver", "skrewed", "skewed"],
            "bolt": ["bolt", "bold", "board", "boat", "bult", "bolte", "bohlt", "boy", "bowl", "boyd", "ball", "pool", "paul", "bullets", "bullet"],
            "wrench": ["wrench", "rench", "range", "french", "spanner", "rensh", "right","branch", "ranch", "wench", "trench"],
            "measuring tape": ["measuring tape", "measuring tap", "tape", "measure", "measuring the", "measuring", "measur", "measure tape"],
            "hammer": ["hammer", "armor", "amor", "ammer", "hamer", "hamr", "mallet", "hummer", "however", "harvard", "homo", "rough"],
            "plier": ["plier", "players", "pliers", "pryer", "player", "plyer", "pincher", "apply", "flyer", "flier", "liar", "lawyer"],
        }
        
        # Initialize Text-to-Speech
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 170)
        self.tts.setProperty('volume', 0.9)
        
        # Vosk model
        self.model = None
        self.recognizer = None
        self.stream = None
        self.p = None
    
    def log(self, message, level="info"):
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(f"[VOICE {level.upper()}] {message}")
    
    def set_command_callback(self, callback):
        self.command_callback = callback
    
    def pause_listening(self):
        """Pause voice listening"""
        self.paused = True
    
    def resume_listening(self):
        """Resume voice listening"""
        self.paused = False
    
    def initialize_voice(self):
        """Initialize voice recognition system"""
        try:
            model_path = "./vosk-model-small-en-us-0.15"
            if not os.path.exists(model_path):
                self.log(f"Vosk model not found at {model_path}", "error")
                return False
            
            self.log("Loading Vosk model...", "info")
            self.model = Model(model_path)
            
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8000,
                input_device_index=3
            )
            
            self.recognizer = KaldiRecognizer(self.model, 16000)
            
            self.log("Voice recognition ready!", "success")
            return True
            
        except Exception as e:
            self.log(f"Failed to initialize voice: {e}", "error")
            return False
    
    def speak(self, text):
        """Make the robot speak through speakers"""
        if not text or self.is_speaking:
            return
        
        self.is_speaking = True
        self.log(f"Speaking: {text}", "info")
        
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except Exception as e:
            self.log(f"Speaking error: {e}", "error")
            # If speech fails, at least print it
            self.log(f"[SPEAKING]: {text}", "info")
            time.sleep(len(text.split()) * 0.3)  # Simulate speaking time
        
        self.is_speaking = False
    
    def get_fetching_message(self, tool):
        """Generate fetching message - ALWAYS USED when fetching any tool"""
        return f"Fetching the {tool} for you!"
    
    def get_confirmation_message(self, tool):
        """Generate confirmation message after fetching"""
        return f"Here is your {tool}! What else can I get for you?"
    
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
    
    def process_voice_command(self, text):
        """Process a voice command - MODIFIED to skip confirmation"""
        text_lower = text.lower()
        
        # Check for help first
        if any(word in text_lower for word in ["help", "what can you do", "instructions"]):
            help_msg = "I can fetch tools for you. Just say the name of the tool, like 'hammer' or 'wrench'."
            self.speak(help_msg)
            return None
        
        # Check for tool request - primary action
        tool = self.find_tool(text_lower)
        
        if tool:
            # MODIFIED: Skip speaking fetching message and return tool directly
            # This allows immediate grabbing without confirmation
            return tool
        
        # Check for greeting if no tool or help command was found
        if any(greet in text_lower for greet in ["hello", "hi", "hey", "greetings"]):
            greeting_msg = "Hello! I'm here to help. Just say the name of a tool you need."
            self.speak(greeting_msg)
            return None
        
        # If nothing was understood
        error_msg = "I didn't catch that. Please say the name of a tool, like 'hammer' or 'screwdriver'..."
        self.speak(error_msg)
        return None
    
    def run(self):
        """Main voice recognition loop - UPDATED to respect paused state"""
        self.running = True
        
        if not self.initialize_voice():
            self.log("Failed to initialize voice recognition", "error")
            return
        
        self.log("Voice recognition started", "success")
        welcome_msg = "Hello! I am your Garage Assistant. I can fetch tools for you. Just say the name of a tool you need, like 'wrench' or 'hammer'..."
        self.speak(welcome_msg)
        
        while self.running and self.voice_enabled:
            try:
                # Skip processing if paused
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                if self.is_speaking:
                    time.sleep(0.1)
                    continue
                
                data = self.stream.read(4000, exception_on_overflow=False)
                
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '').strip()
                    
                    if text and len(text) > 2:
                        current_time = time.time()
                        if current_time - self.last_command_time > 1.5:
                            tool = self.process_voice_command(text)
                            if tool and self.command_callback:
                                self.command_callback(tool)
                            self.last_command_time = current_time
                
                time.sleep(0.01)
                
            except Exception as e:
                self.log(f"Voice recognition error: {e}", "error")
                time.sleep(1)
    
    def stop(self):
        """Stop voice recognition"""
        self.running = False
        self.voice_enabled = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.p:
            self.p.terminate()
        
        self.log("Voice recognition stopped", "info")