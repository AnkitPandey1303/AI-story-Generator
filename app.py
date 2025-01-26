import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import os
import cv2
import ffmpeg

# Load Stable Diffusion Model
@st.cache_resource
def load_model():
    return StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda" if torch.cuda.is_available() else "cpu")

pipe = load_model()

# Create output directory in the temporary folder
output_folder = "/tmp/story_frames"
os.makedirs(output_folder, exist_ok=True)

# Streamlit App
st.title("📖 AI Story-to-Video Generator 🎥")

# Story Input
story_text = st.text_area("Enter Your Story:", "Once upon a time in a magical forest, a small dragon found a glowing crystal...")

if st.button("Generate Story Video"):
    if story_text.strip():
        st.write("Generating Frames...")

        # Split story into scenes based on sentences or periods
        story_scenes = story_text.split(". ")

        # Generate Frames
        frame_paths = []
        for i, scene in enumerate(story_scenes):
            if scene.strip():  # Ignore empty scenes
                text_description = f"Scene {i+1}: {scene.strip()}"
                image = pipe(text_description).images[0]

                # Save image in temporary directory
                frame_path = os.path.join(output_folder, f"frame_{i+1}.png")
                image.save(frame_path)
                frame_paths.append(frame_path)

                # Display Image
                st.image(frame_path, caption=f"Scene {i+1}")

        # Create Video from Frames
        st.write("Creating Video from Frames...")

        # Load first image to get dimensions
        first_frame = cv2.imread(frame_paths[0])
        height, width, _ = first_frame.shape

        video_filename = "/tmp/story_video.avi"
        video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), 1, (width, height))  # Frame rate is set to 1 frame per second

        for frame in frame_paths:
            img = cv2.imread(frame)
            video.write(img)

        video.release()

        # Convert to MP4 using FFmpeg
        mp4_filename = "/tmp/story_video.mp4"
        ffmpeg.input(video_filename).output(mp4_filename).run(overwrite_output=True)

        # Provide Download Link
        with open(mp4_filename, "rb") as f:
            st.download_button("Download Story Video", f, file_name="story_video.mp4", mime="video/mp4")
    else:
        st.warning("Please enter a story.")
