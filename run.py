"""
Entry point for Deep Research Agent.
Run with: python run.py
"""
import os
from dotenv import load_dotenv

load_dotenv()

from src.app import demo

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
