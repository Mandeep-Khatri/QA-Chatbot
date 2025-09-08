"""Setup script for the Q&A chatbot package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="qa-chatbot",
    version="1.0.0",
    author="Mandeep",
    author_email="mandeeppaudel00@gmail.com",
    description="Gemini by Mandy - Q&A chatbot for course materials using Gemini 1.5 Flash and LangChain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mandeep-Khatri/QA-Chatbot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "black>=23.11.0",
            "flake8>=6.1.0",
        ],
        "web": [
            "streamlit>=1.28.1",
            "gradio>=4.7.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "qa-chatbot=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
)
