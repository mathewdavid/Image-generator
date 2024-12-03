import streamlit as st
from PIL import Image
import io
import os
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

# Stability AI setup
os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'


def validate_api_key(api_key):
    """Validate the API key by attempting to create a client."""
    try:
        client.StabilityInference(key=api_key, verbose=False)
        return True
    except Exception:
        return False


def generate_image(prompt, api_key):
    """Generate an image using Stable Diffusion via Stability AI API."""
    try:
        # Set up our connection to the API.
        stability_api = client.StabilityInference(
            key=api_key,  # API Key
            verbose=True,  # Print debug messages
            engine="stable-diffusion-xl-1024-v1-0",  # Set the engine to use for generation
        )

        # Set up our initial generation parameters.
        answers = stability_api.generate(
            prompt=prompt,
            seed=992446758,  # If a seed is provided, it will be used for generation
            steps=50,  # Amount of inference steps performed on image generation
            cfg_scale=8.0,  # Influences how strongly your generation is guided to match your prompt
            width=1024,  # Generation width, defaults to 512 if not included
            height=1024,  # Generation height, defaults to 512 if not included
            samples=1,  # Number of images to generate, defaults to 1 if not included
            sampler=generation.SAMPLER_K_DPMPP_2M  # Choose which sampler we want to denoise our generation with
        )

        # Iterate over the generated images and return the first one
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    st.warning("Your request activated the API's safety filters and could not be processed.")
                    return None
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))
                    return img

    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None


def main_screen(api_key):
    st.title("Stable Diffusion Image Generator")

    # User input
    prompt = st.text_area("Enter your image prompt:", height=100)

    # Generate button
    if st.button("Generate Image"):
        if prompt:
            with st.spinner("Generating image with Stable Diffusion..."):
                generated_image = generate_image(prompt, api_key)
                if generated_image:
                    st.image(generated_image, caption="Generated Image by Stable Diffusion", use_container_width=True)
                else:
                    st.error("Failed to generate image. Please try a different prompt.")
        else:
            st.warning("Please enter a prompt before generating.")

    # Instructions and examples
    st.sidebar.header("Instructions")
    st.sidebar.write("""
    1. Enter a descriptive prompt for the image you want to generate.
    2. Click the 'Generate Image' button.
    3. Wait for Stable Diffusion to create and display your image.
    """)

    st.sidebar.header("Example Prompts")
    st.sidebar.write("""
    - A serene landscape with a mountain lake at sunset
    - A futuristic robot in a cyberpunk city
    - A whimsical fairy tale castle in a magical forest
    - An abstract representation of love and harmony
    """)

    # Add information about the app
    st.sidebar.header("About")
    st.sidebar.write("""
    This app uses Stable Diffusion via the Stability AI API to generate images based on text prompts. 
    It's designed to assist in creating unique visual content for various creative purposes.
    """)


def main():
    st.set_page_config(page_title="Stable Diffusion Image Generator", layout="wide")

    # Check if API key is already in session state
    if 'api_key' not in st.session_state or not st.session_state.api_key:
        st.title("Welcome to Stable Diffusion Image Generator")
        st.write("Please enter your Stability AI API key to continue.")

        api_key = st.text_input("Enter your Stability AI API key:", type="password")
        if st.button("Validate and Continue"):
            if validate_api_key(api_key):
                st.session_state.api_key = api_key
                st.success("API key validated successfully!")
                st.empty()  # Clear the current content
                main_screen(api_key)  # Directly call main_screen with the validated API key
            else:
                st.error("Invalid API key. Please try again.")
    else:
        main_screen(st.session_state.api_key)


if __name__ == "__main__":
    main()

