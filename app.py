import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import PIL.Image

st.set_page_config(layout="wide")
st.title("Stable Diffusion Image Generator")

# Define the model ID from your notebook
model_id = "dreamlike-art/dreamlike-diffusion-1.0"

@st.cache_resource
def load_model():
    """Loads the Stable Diffusion pipeline and caches it."""
    # Ensure we use float16 for memory efficiency and speed
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    # Move model to CUDA if available, otherwise CPU (though CUDA is highly recommended)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        st.warning("CUDA not available. Running on CPU, which will be significantly slower.")
        pipe = pipe.to("cpu")
    return pipe

# Load the model
pipe = load_model()

st.header("Generate an Image")

# Input fields for prompt and parameters
prompt = st.text_area(
    "Enter your image prompt:",
    value="BMW M4 Competition G82, matte black with blue neon underglow, futuristic cyberpunk city, heavy rain, neon signs, reflective wet roads, cinematic composition, volumetric lighting, smoke, glowing taillights, carbon fiber details, ultra realistic, dramatic contrast, professional car photography, 8K, masterpiece, extremely detailed, ray tracing, Unreal Engine 5"
)

negative_prompt = st.text_input(
    "Enter a negative prompt (optional, e.g., 'ugly, low quality'):",
    value="ugly,low quality,distorted"
)

num_inference_steps = st.slider(
    "Number of inference steps:",
    min_value=10, max_value=150, value=100, step=5
)

num_images_per_prompt = st.number_input(
    "Number of images to generate:",
    min_value=1, max_value=4, value=1, step=1
)

# Generate button
if st.button("Generate Image(s)"):
    if not prompt:
        st.error("Please enter a prompt to generate an image.")
    else:
        with st.spinner("Generating image(s)... This might take a moment."):
            params = {
                'num_inference_steps': num_inference_steps,
                'negative_prompt': negative_prompt if negative_prompt else None,
                'num_images_per_prompt': num_images_per_prompt
            }
            
            # Generate images
            images = pipe(prompt, **params).images
            
            # Display images
            st.success("Image(s) Generated!")
            
            if len(images) == 1:
                st.image(images[0], caption=prompt, use_column_width=True)
            else:
                cols = st.columns(len(images))
                for i, img in enumerate(images):
                    with cols[i]:
                        st.image(img, caption=f"Image {i+1}", use_column_width=True)

st.markdown("---")
st.info("Powered by Hugging Face Diffusers and Streamlit")
