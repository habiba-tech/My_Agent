
import pyttsx3

# Initialize Jarvis voice
engine = pyttsx3.init()

# Set speaking speed
engine.setProperty("rate", 165)

# Set volume
engine.setProperty("volume", 1.0)

# Make Jarvis speak
engine.say("Hello Habiba. I am Jarvis.")
engine.say("My voice system is now working.")
engine.runAndWait()

print("Voice test completed!")
