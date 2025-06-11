#!/usr/bin/env python3
"""
Video Generator for API Database
Generates videos for all stories in the api/db.db database using background videos,
text-to-speech, and on-screen text overlays.
"""

import os
import random
import sqlite3
import time
import textwrap
from pathlib import Path

from gtts import gTTS
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, AudioFileClip
from moviepy.config import change_settings

change_settings({"IMAGEMAGICK_BINARY": "/opt/homebrew/bin/magick"})

class APIVideoGenerator:
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.api_dir = self.script_dir.parent / "api"
        self.backgrounds_dir = self.script_dir / "generatevideos" / "backgrounds"
        self.output_dir = self.script_dir / "generatevideos" / "videos"
        self.temp_dir = self.script_dir / "temp"
        self.db_path = self.api_dir / "db.db" 
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Video settings
        self.max_chars_per_line = 30  
        self.max_lines_on_screen = 4  
        self.fontsize = 64  
        self.font_color = 'white'
        self.font_stroke_color = None   
        self.font_stroke_width = 0  
        
    def get_all_stories(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check database structure first
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"Database tables: {[t[0] for t in tables]}")
            
            # Get stories with labels
            cursor.execute('''
                SELECT v.videoId, v.text, GROUP_CONCAT(l.label, ', ') as labels
                FROM videos v
                LEFT JOIN labels l ON v.videoId = l.videoId
                GROUP BY v.videoId, v.text
                ORDER BY v.videoId
            ''')
            
            stories = cursor.fetchall()
            conn.close()
            
            print(f"Found {len(stories)} stories in database")
            return stories
            
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return []
    
    def get_background_videos(self):
        """Get list of available background videos"""
        if not self.backgrounds_dir.exists():
            print(f"Background directory not found: {self.backgrounds_dir}")
            print("Please create the directory and add .mp4 background videos")
            raise FileNotFoundError(f"Background directory not found: {self.backgrounds_dir}")
        
        background_files = list(self.backgrounds_dir.glob("*.mp4"))
        if not background_files:
            print(f"No background videos found in: {self.backgrounds_dir}")
            print("Please add .mp4 background videos to this directory")
            raise FileNotFoundError("No background videos found (.mp4 files)")
        
        print(f"Found {len(background_files)} background videos:")
        for bg in background_files:
            print(f"  - {bg.name}")
        return background_files
    
    def wrap_text_for_video(self, text):
        # Clean the text first
        text = text.strip()
        
        # Split into sentences for better readability
        sentences = text.replace('. ', '.\n').split('\n')
        wrapped_lines = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 0:
                # Wrap long sentences
                wrapped = textwrap.fill(sentence.strip(), width=self.max_chars_per_line)
                wrapped_lines.extend(wrapped.split('\n'))
        
        return '\n'.join(wrapped_lines)
    
    def create_text_clips(self, text, duration):
        wrapped_text = self.wrap_text_for_video(text)
        lines = wrapped_text.split('\n')
        
        # Split text into chunks of max 8 words per screen
        all_words = wrapped_text.replace('\n', ' ').split()
        word_chunks = []
        
        # Group words into chunks of 8 words maximum
        for i in range(0, len(all_words), 8):
            chunk = ' '.join(all_words[i:i+8])
            word_chunks.append(chunk)
        
        text_clips = []
        
        if len(word_chunks) == 1:
            # All text fits in one screen
            text_clip = TextClip(
                word_chunks[0],
                fontsize=self.fontsize,
                color=self.font_color,
                font='Comic-Sans-MS',
                stroke_color=self.font_stroke_color,
                stroke_width=self.font_stroke_width,
                method='caption'
            ).set_position('center').set_duration(duration)
            
            text_clips.append(text_clip)
        else:
            # Split text into multiple screens (8 words each)
            clip_duration = duration / len(word_chunks)
            
            for i, chunk in enumerate(word_chunks):
                start_time = i * clip_duration
                
                text_clip = TextClip(
                    chunk,
                    fontsize=self.fontsize,
                    color=self.font_color,
                    font='Comic-Sans-MS',
                    stroke_color=self.font_stroke_color,
                    stroke_width=self.font_stroke_width,
                    method='caption'
                ).set_position('center').set_start(start_time).set_duration(clip_duration)
                
                text_clips.append(text_clip)
        
        return text_clips
    
    def clean_text_for_tts(self, text):
        """Clean text for TTS by removing special characters and formatting"""
        import re
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove numbers that are not part of words
        text = re.sub(r'\b\d+\b', '', text)
        
        return text.strip()

    def generate_audio(self, text, video_id):
        try:
            # Clean text for TTS (but don't limit word count)
            clean_text = self.clean_text_for_tts(text)
            
            if not clean_text:
                print(f"  ⚠️  Empty text for video {video_id}")
                return None, 0
            
            print(f"TTS text: '{clean_text[:50]}...' ({len(clean_text.split())} words)")
            
            tts = gTTS(text=clean_text, lang='en', slow=False)
            audio_path = self.temp_dir / f"audio_{video_id}_{int(time.time())}.mp3"
            tts.save(str(audio_path))
            
            # Get audio duration
            audio = AudioSegment.from_file(str(audio_path))
            duration = len(audio) / 1000.0  # Convert to seconds
            
            return str(audio_path), duration
            
        except Exception as e:
            print(f"Error generating audio for video {video_id}: {e}")
            return None, 0
    
    def process_background_video(self, background_path, target_duration):
        try:
            video = VideoFileClip(str(background_path))
            original_duration = video.duration
            
            if original_duration < target_duration:
                # Loop video if it's shorter than needed
                loops = int(target_duration // original_duration) + 1
                video = concatenate_videoclips([video] * loops)
            
            # Trim to exact duration
            video = video.subclip(0, target_duration)
            return video
            
        except Exception as e:
            print(f"Error processing background video: {e}")
            return None
    
    def generate_video(self, video_id, story_text, labels, background_files):
        try:
            print(f"Generating video {video_id}...")
            
            # Check if video already exists
            output_path = self.output_dir / f"{video_id}.mp4"
            if output_path.exists():
                print(f"Video {video_id} already exists, skipping...")
                return True
            
            # Validate story text
            if not story_text or not story_text.strip():
                print(f"Empty story text for video {video_id}, skipping...")
                return False
            
            # Select random background
            background_path = random.choice(background_files)
            print(f"Using background: {background_path.name}")
            
            # Generate audio
            audio_path, audio_duration = self.generate_audio(story_text, video_id)
            if not audio_path or audio_duration <= 0:
                print(f"Failed to generate audio for video {video_id}")
                return False
            
            print(f"Audio duration: {audio_duration:.2f} seconds")
            
            # Ensure minimum duration
            if audio_duration < 3:
                audio_duration = 5  # Minimum 5 seconds
                print(f"Extended duration to {audio_duration} seconds")
            
            # Process background video
            background_video = self.process_background_video(background_path, audio_duration)
            if not background_video:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                return False
            
            # Create text overlays
            text_clips = self.create_text_clips(story_text, audio_duration)
            
            # Load audio
            audio_clip = AudioFileClip(audio_path)
            
            # Combine all elements
            final_clips = [background_video] + text_clips
            final_video = CompositeVideoClip(final_clips)
            final_video = final_video.set_audio(audio_clip)
            
            # Write video file
            print(f"Writing video to: {output_path.name}")
            final_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                fps=24,
                verbose=False,
                logger=None
            )
            
            # Cleanup
            final_video.close()
            background_video.close()
            audio_clip.close()
            for clip in text_clips:
                clip.close()
            
            # Remove temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            print(f"Video {video_id} generated successfully!")
            return True
            
        except Exception as e:
            print(f"Error generating video {video_id}: {e}")
            # Cleanup temporary files
            if 'audio_path' in locals() and audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            return False
    
    def generate_all_videos(self, limit=None, start_from=1, skip_existing=True):
        """Generate videos for all stories in the database"""
        print("Starting video generation for API database stories...")
        print("=" * 60)
        
        # Get stories and backgrounds
        stories = self.get_all_stories()
        if not stories:
            print("No stories found in database!")
            return
        
        try:
            background_files = self.get_background_videos()
        except FileNotFoundError as e:
            print(f"{e}")
            return
        
        # Apply filters
        if start_from > 1:
            stories = [s for s in stories if s[0] >= start_from]
        
        if limit:
            stories = stories[:limit]
        
        print(f"Processing {len(stories)} stories...")
        print(f"Output directory: {self.output_dir}")
        print(f"Background videos: {len(background_files)}")
        print(f"Skip existing: {skip_existing}")
        print("=" * 60)
        
        successful = 0
        failed = 0
        skipped = 0
        
        for i, (video_id, story_text, labels) in enumerate(stories, 1):
            print(f"\n[{i}/{len(stories)}] Processing story {video_id}...")
            
            if skip_existing and (self.output_dir / f"{video_id}.mp4").exists():
                print(f" Video {video_id} already exists, skipping...")
                skipped += 1
                continue
            
            success = self.generate_video(video_id, story_text, labels, background_files)
            
            if success:
                successful += 1
            else:
                failed += 1
            
            # Progress update
            if i % 10 == 0:
                print(f"\nProgress: {i}/{len(stories)} processed")
                print(f"Success: {successful}, Failed: {failed}, Skipped: {skipped}")
        
        print("\n" + "=" * 60)
        print(f"Generation Complete!")
        print(f"Final Summary:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Skipped: {skipped}")
        print(f"  Output directory: {self.output_dir}")
        
        # Copy videos to client folder
        if successful > 0:
            self.copy_videos_to_client()
        
        if failed > 0:
            print(f"\n{failed} videos failed to generate. Check the errors above.")
        elif successful > 0:
            print(f"\n All videos generated successfully!")
            print(f" You can now use these videos with your TikTok-like frontend!")

    def copy_videos_to_client(self):
        """Copy generated videos to client/public/videos folder"""
        import shutil
        
        client_videos_dir = self.script_dir.parent / "client" / "public" / "videos"
        client_videos_dir.mkdir(parents=True, exist_ok=True)
        
        copied_count = 0
        video_files = list(self.output_dir.glob("*.mp4"))
        
        print(f"\nCopying {len(video_files)} videos to client folder...")
        print(f"Target: {client_videos_dir}")
        
        for video_file in video_files:
            target_path = client_videos_dir / video_file.name
            try:
                shutil.copy2(video_file, target_path)
                copied_count += 1
                if copied_count % 10 == 0:
                    print(f"Copied {copied_count}/{len(video_files)} videos...")
            except Exception as e:
                print(f"Error copying {video_file.name}: {e}")
        
        print(f"Successfully copied {copied_count} videos to client folder!")
        return copied_count

def main():
    """Main function with command line options"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate videos for all stories in API database")
    parser.add_argument("--limit", type=int, help="Limit number of videos to generate")
    parser.add_argument("--start-from", type=int, default=1, help="Start from specific video ID")
    parser.add_argument("--test", action="store_true", help="Generate only first 3 videos as test")
    parser.add_argument("--force", action="store_true", help="Regenerate existing videos")
    
    args = parser.parse_args()
    
    if args.test:
        args.limit = 3
        print("Test mode: Generating only the first 3 videos")
    
    generator = APIVideoGenerator()
    generator.generate_all_videos(
        limit=args.limit, 
        start_from=args.start_from,
        skip_existing=not args.force
    )

if __name__ == "__main__":
    main()
