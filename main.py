from IPython.display import display, Image, Audio

import cv2
import base64
import time
import openai
import os
import requests
import numpy as np
import pygame
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

video = cv2.VideoCapture("data/tenz_clip.mp4")
# Initialize the mixer module
pygame.mixer.init()

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    # print(frame)
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(len(base64Frames), "frames read.")

PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "These are the frames of a valorant clip from a tournament. You are a valorant shoutcaster who has casted multiple tournaments. Create a voiceover script with the lingo of a valorant caster throughout the frames given. Only include the commentary. Avoid unnecessary sentences you provide besides the content of the frames provided.",
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::648]),
        ],
    },
]
params = {
    "model": "gpt-4-vision-preview",
    "messages": PROMPT_MESSAGES,
    "api_key": openai.api_key,
    "headers": {"Openai-Version": "2020-11-07"},
    "max_tokens": 200,
}

# result = openai.ChatCompletion.create(**params)
# print(result.choices[0].message.content)

# response = requests.post(
#     "https://api.openai.com/v1/audio/speech",
#     headers={
#         "Authorization": f"Bearer {openai.api_key}",
#     },
#     json={
#         "model": "tts-1",
#         "input": result.choices[0].message.content,
#         "voice": "onyx",
#     },
# )

# audio_data = b""
# for chunk in response.iter_content(chunk_size=1024 * 1024):
#     audio_data += chunk

# # Save the audio data to a file
# with open('finalized_clip.wav', 'wb') as f:
#     f.write(audio_data)

# Load the audio file
# sound = pygame.mixer.Sound('finalized_clip.wav')

# # Start playing the audio
# sound.play()

# # This loop iterates over each base64 encoded frame in the base64Frames list
# for img in base64Frames:
#     # The base64 encoded frame is decoded back into bytes
#     img_data = base64.b64decode(img.encode("utf-8"))
#     # The byte data is converted into a numpy array of type unsigned int 8
#     nparr = np.frombuffer(img_data, np.uint8)
#     # The numpy array is decoded into an image
#     img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     # The image is displayed
#     cv2.imshow('valorant clip', img_np)
#     # The program waits for 25 ms before moving to the next frame
#     cv2.waitKey(25)

# # This line is necessary to close the windows at the end of the program
# cv2.destroyAllWindows()

# # Stop the audio at the end
# sound.stop()


# print(chat_with_openai("What is your name?"))
