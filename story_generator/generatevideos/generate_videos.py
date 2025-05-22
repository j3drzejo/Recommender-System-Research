import os
import random
import time
from gtts import gTTS
from pydub import AudioSegment
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips

# Set these variables manually
story = "Your story text goes here. This will be converted to speech and subtitles."
video_id = "1"

# Path configuration
backgrounds_dir = "story_generator/generatevideos/backgrounds"
output_dir = "story_generator/generatevideos/videos"
temp_dir = "temp"

# Create directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

# Select random background video
background_files = [f for f in os.listdir(backgrounds_dir) if f.endswith(".mp4")]
if not background_files:
    raise Exception("No background videos found in the backgrounds directory")
selected_bg = os.path.join(backgrounds_dir, random.choice(background_files))

# Generate text-to-speech
tts = gTTS(text=story, lang='en')
tts_path = os.path.join(temp_dir, f"temp_{time.time()}.mp3")
tts.save(tts_path)

# Get audio duration
audio = AudioSegment.from_file(tts_path)
audio_duration = len(audio) / 1000  # Convert to seconds

# Process video
video = VideoFileClip(selected_bg)
original_duration = video.duration

# If video is shorter than needed, loop it
if original_duration < audio_duration:
    loops = int(audio_duration // original_duration) + 1
    video = concatenate_videoclips([video] * loops)

# Trim video to match audio duration exactly
video = video.subclip(0, audio_duration)

# Create subtitle clip
subtitle = TextClip(story, fontsize=40, color='white', 
                   font='Arial-Bold', method='caption',
                   size=(video.w * 0.9, None), align='center')
subtitle = subtitle.set_position(('center', 'center')).set_duration(audio_duration)

# Create final video with subtitles
final_video = CompositeVideoClip([video, subtitle])

# Add audio to video
final_video = final_video.set_audio(tts_path)

# Write output file
output_path = os.path.join(output_dir, f"{video_id}.mp4")
final_video.write_videofile(output_path, codec='libx264', 
                           audio_codec='aac', 
                           fps=video.fps)

# Cleanup temporary files
os.remove(tts_path)