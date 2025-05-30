from setuptools import setup, find_packages

setup(
    name="soap-rag-generator",
    version="1.0.0",
    description="RAG-based SOAP Note Generator using LangChain",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-openai>=0.0.5",
        "langchain-community>=0.0.10",
        "faiss-cpu>=1.7.4",
        "chromadb>=0.4.22",
        "sentence-transformers>=2.2.2",
        "transformers>=4.36.2",
        "torch>=2.1.2",
        "datasets>=2.16.1",
        "pandas>=2.1.4",
        "numpy>=1.24.4",
        "rouge-score>=0.1.2",
        "nltk>=3.8.1",
        "evaluate>=0.4.1",
        "streamlit>=1.29.0",
        "plotly>=5.17.0",
        "openai>=1.6.1",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
