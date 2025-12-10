import json
import pyaudio
import time
import os
import sys
import re
import threading
import pyttsx3
from vosk import Model, KaldiRecognizer

class AlwaysListeningAssistant:
    def __init__(self):
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
        print("Initializing speech engine...")
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 170)
        self.tts.setProperty('volume', 0.9)
        
        # Robot settings
        self.robot_name = "Arrange Assistant"
        self.is_speaking = False
        self.command_queue = []
        self.last_command_time = 0
        
        # Load Vosk model
        model_path = "./vosk-model-small-en-us-0.15"
        self.model = Model(model_path)
        print("✅ Speech recognition ready!")
    
    def speak(self, text):
        """Make the robot speak through speakers - SIMPLIFIED"""
        if not text:
            return
            
        print(f"\n🤖 {self.robot_name}: {text}")
        
        # Mark robot as speaking
        self.is_speaking = True
        self.tts.say(text)
        self.tts.runAndWait()
        try:
            # Speak the text (this is what makes the sound)
            self.tts.say(text)
            self.tts.runAndWait()  # This blocks until speech is complete
        except Exception as e:
            print(f"❌ Speaking error: {e}")
            # If speech fails, at least print it
            print(f"[SPEAKING]: {text}")
            time.sleep(len(text.split()) * 0.3)  # Simulate speaking time
        
        # Mark robot as done speaking
        self.is_speaking = False
        print("✅ Robot finished speaking")
    
    def get_fetching_message(self, tool):
        """Generate fetching message - ALWAYS USED when fetching any tool"""
        # Always use this exact format
        return f"Fetching the {tool} for you!"
    
    def get_confirmation_message(self, tool):
        """Generate confirmation message after fetching"""
        return f"Here is your {tool}! What else can I get for you?"
    
    def listen_continuously(self):
        """Always listen in background"""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8000
        )
        
        recognizer = KaldiRecognizer(self.model, 16000)
        
        print("🔊 Always listening...")
        
        while True:
            try:
                # Don't process audio while robot is speaking
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
                            self.command_queue.append(text)
                            self.last_command_time = current_time
                
            except KeyboardInterrupt:
                break
            except:
                pass
    
    def find_tool(self, text):
        """Find which tool was mentioned - SIMPLIFIED"""
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
        """Process a user command - SIMPLIFIED to always speak fetching"""
        print(f"\n🎤 User said: '{text}'")
        
        text_lower = text.lower()
        
        # Check for greeting
        if any(greet in text_lower for greet in ["hello", "hi", "hey", "greetings"]):
            self.speak("Hello! I'm here to help. What tool would you like me to fetch?")
            return True
        
        # Check for help
        if any(word in text_lower for word in ["help", "what can you do", "instructions"]):
            self.speak("I can fetch tools for you. Just say 'get me' followed by the tool name. Try 'get me the hammer' or 'fetch the screwdriver'.")
            return True
        
        # Check for tool request
        tool = self.find_tool(text_lower)
        
        if tool:
            command_words = ["get", "bring", "give", "pass", "hand", "fetch", "need", "want", "grab"]
            has_command = any(cmd in text_lower for cmd in command_words)
            
            if has_command:
                # ALWAYS speak fetching message
                fetching_msg = self.get_fetching_message(tool)
                self.speak(fetching_msg)
                
                # Simulate fetching action
                print(f"\n⚙️ [ACTION] Robot is physically fetching: {tool}")
                time.sleep(1)  # Simulate time to fetch
                
                # Confirm completion
                confirmation_msg = self.get_confirmation_message(tool)
                self.speak(confirmation_msg)
                return True
            else:
                # Tool mentioned but no command - still acknowledge
                self.speak(f"I heard '{tool}'. To get it, say 'get me the {tool}'.")
                return True
        
        # No tool found
        self.speak("I didn't understand. Try saying: 'get me the wrench', 'fetch the hammer', or 'I need a screwdriver'.")
        return True
    
    def run(self):
        """Main function - always listening"""
        print("\n" + "="*60)
        print(f"🛠️ {self.robot_name.upper()} - ALWAYS LISTENING")
        print("="*60)
        print("\nThe robot will speak ALL responses, including fetching messages.")
        print("="*60 + "\n")
        
        # Start background listening thread
        listen_thread = threading.Thread(target=self.listen_continuously, daemon=True)
        listen_thread.start()
        
        # Give initial welcome message
        welcome_msg = "Hello! I am your Arrange Assistant. I can fetch tools for you. Try saying 'get me the wrench' or 'fetch the hammer'."
        self.speak(welcome_msg)
        
        # Main loop
        print("\n💡 Ready! Speak your command...")
        print("-" * 60)
        
        while True:
            try:
                # Process commands from queue
                if self.command_queue:
                    text = self.command_queue.pop(0)
                    self.process_command(text)
                
                # If idle for 20 seconds, give a reminder
                if time.time() - self.last_command_time > 20 and not self.is_speaking:
                    self.speak("I'm listening. You can ask for tools anytime.")
                    self.last_command_time = time.time()
                
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                print("\n👋 Shutting down...")
                self.speak("Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

# Test function to verify speech works
def test_speech():
    """Test if speech is working"""
    print("Testing speech output...")
    try:
        tts = pyttsx3.init()
        print("✅ Speech engine working")
        
        test_messages = [
            "Testing, testing, one two three.",
            "Fetching the wrench for you!",
            "Here is your hammer!"
        ]
        
        for msg in test_messages:
            print(f"\nSpeaking: {msg}")
            tts.say(msg)
            tts.runAndWait()
            time.sleep(1)
            
        print("\n✅ Speech test successful!")
        return True
    except Exception as e:
        print(f"❌ Speech test failed: {e}")
        return False

# Run it directly - no choices, always listening mode
if __name__ == "__main__":
    print("Starting Arrange Assistant...")
    
    # First test speech
    if not test_speech():
        print("\n⚠️  Speech may not work. Make sure:")
        print("1. Speakers are turned on")
        print("2. Volume is up")
        print("3. Try: pip install pyttsx3")
    
    # Create and run the assistant
    assistant = AlwaysListeningAssistant()
    assistant.run()