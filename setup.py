from setuptools import setup, find_packages

setup(
    name="ai-image-captioning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.18.0",
        "gradio>=3.0.0",
        "Pillow>=8.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered image captioning app using BLIP and GPT-2",
)