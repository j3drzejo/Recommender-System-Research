#!/usr/bin/env python3
"""
Video Generator for All Stories
Generates videos for all stories in the database using background videos,
text-to-speech, and on-screen text overlays.
"""

import os
import random
import sqlite3
import time
import textwrap
from pathlib import Path

try:
    from gtts import gTTS
    from pydub import AudioSegment
    from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, AudioFileClip
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install dependencies with:")
    print("pip install gtts pydub moviepy")
    exit(1)

class VideoGenerator:
    def __init__(self):
        # Path configuration
        self.script_dir = Path(__file__).parent
        self.backgrounds_dir = self.script_dir / "generatevideos" / "backgrounds"
        self.output_dir = self.script_dir / "generatevideos" / "videos"
        self.temp_dir = self.script_dir / "temp"
        self.db_path = self.script_dir / "db.db"
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Video settings
        self.max_chars_per_line = 50
        self.max_lines_on_screen = 8
        self.fontsize = 36
        self.font_color = 'white'
        self.font_stroke_color = 'black'
        self.font_stroke_width = 2
        
    def get_all_stories(self):
        """Retrieve all stories from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
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
            raise FileNotFoundError(f"Background directory not found: {self.backgrounds_dir}")
        
        background_files = list(self.backgrounds_dir.glob("*.mp4"))
        if not background_files:
            raise FileNotFoundError("No background videos found (.mp4 files)")
        
        print(f"Found {len(background_files)} background videos")
        return background_files
    
    def wrap_text_for_video(self, text):
        """Wrap text to fit nicely on screen"""
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
        """Create text clips with proper formatting and timing"""
        wrapped_text = self.wrap_text_for_video(text)
        lines = wrapped_text.split('\n')
        
        # If text is too long, create multiple clips
        text_clips = []
        words_per_minute = 150  # Average reading speed
        
        if len(lines) <= self.max_lines_on_screen:
            # Text fits in one screen
            text_clip = TextClip(
                wrapped_text,
                fontsize=self.fontsize,
                color=self.font_color,
                font='Arial-Bold',
                stroke_color=self.font_stroke_color,
                stroke_width=self.font_stroke_width,
                method='caption'
            ).set_position('center').set_duration(duration)
            
            text_clips.append(text_clip)
        else:
            # Split text into multiple screens
            chunk_size = self.max_lines_on_screen
            chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
            clip_duration = duration / len(chunks)
            
            for i, chunk in enumerate(chunks):
                chunk_text = '\n'.join(chunk)
                start_time = i * clip_duration
                
                text_clip = TextClip(
                    chunk_text,
                    fontsize=self.fontsize,
                    color=self.font_color,
                    font='Arial-Bold',
                    stroke_color=self.font_stroke_color,
                    stroke_width=self.font_stroke_width,
                    method='caption'
                ).set_position('center').set_start(start_time).set_duration(clip_duration)
                
                text_clips.append(text_clip)
        
        return text_clips
    
    def generate_audio(self, text, video_id):
        """Generate text-to-speech audio"""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            audio_path = self.temp_dir / f"audio_{video_id}_{time.time()}.mp3"
            tts.save(str(audio_path))
            
            # Get audio duration
            audio = AudioSegment.from_file(str(audio_path))
            duration = len(audio) / 1000.0  # Convert to seconds
            
            return str(audio_path), duration
            
        except Exception as e:
            print(f"Error generating audio for video {video_id}: {e}")
            return None, 0
    
    def process_background_video(self, background_path, target_duration):
        """Process background video to match target duration"""
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
        """Generate a single video from story text"""
        try:
            print(f"Generating video {video_id}...")
            
            # Select random background
            background_path = random.choice(background_files)
            print(f"  Using background: {background_path.name}")
            
            # Generate audio
            audio_path, audio_duration = self.generate_audio(story_text, video_id)
            if not audio_path:
                return False
            
            print(f"  Audio duration: {audio_duration:.2f} seconds")
            
            # Process background video
            background_video = self.process_background_video(background_path, audio_duration)
            if not background_video:
                return False
            
            # Create text overlays
            text_clips = self.create_text_clips(story_text, audio_duration)
            
            # Load audio
            audio_clip = AudioFileClip(audio_path)
            
            # Combine all elements
            final_clips = [background_video] + text_clips
            final_video = CompositeVideoClip(final_clips)
            final_video = final_video.set_audio(audio_clip)
            
            # Output path
            output_path = self.output_dir / f"story_{video_id}.mp4"
            
            # Write video file
            print(f"  Writing video to: {output_path}")
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
            os.remove(audio_path)
            
            print(f"  ✅ Video {video_id} generated successfully!")
            return True
            
        except Exception as e:
            print(f"  ❌ Error generating video {video_id}: {e}")
            # Cleanup temporary files
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)
            return False
    
    def generate_all_videos(self, limit=None, start_from=1):
        """Generate videos for all stories in the database"""
        print("🎬 Starting video generation for all stories...")
        print("=" * 50)
        
        # Get stories and backgrounds
        stories = self.get_all_stories()
        if not stories:
            print("No stories found in database!")
            return
        
        background_files = self.get_background_videos()
        
        # Apply limit and start_from filters
        if start_from > 1:
            stories = [s for s in stories if s[0] >= start_from]
        
        if limit:
            stories = stories[:limit]
        
        print(f"Generating videos for {len(stories)} stories...")
        print(f"Output directory: {self.output_dir}")
        print("=" * 50)
        
        successful = 0
        failed = 0
        
        for video_id, story_text, labels in stories:
            success = self.generate_video(video_id, story_text, labels, background_files)
            
            if success:
                successful += 1
            else:
                failed += 1
            
            print()  # Empty line for readability
        
        print("=" * 50)
        print(f"📊 Generation Summary:")
        print(f"  ✅ Successful: {successful}")
        print(f"  ❌ Failed: {failed}")
        print(f"  📁 Output directory: {self.output_dir}")
        
        if failed > 0:
            print(f"\n⚠️  {failed} videos failed to generate. Check the errors above.")
        else:
            print(f"\n🎉 All videos generated successfully!")

def main():
    """Main function with command line options"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate videos for all stories in database")
    parser.add_argument("--limit", type=int, help="Limit number of videos to generate")
    parser.add_argument("--start-from", type=int, default=1, help="Start from specific video ID")
    parser.add_argument("--test", action="store_true", help="Generate only first video as test")
    
    args = parser.parse_args()
    
    if args.test:
        args.limit = 1
        print("🧪 Test mode: Generating only the first video")
    
    generator = VideoGenerator()
    generator.generate_all_videos(limit=args.limit, start_from=args.start_from)

if __name__ == "__main__":
    main()
