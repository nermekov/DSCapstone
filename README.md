Основные файлы c исследованием это
- DAC_Model_training.ipynb
- Data_generation.ipynb
- EDA.ipynb  
Среда для обучения Google colab, машина L4

Демо запускается через app.py(streamit)
можно как через докер, так и через Windows Subsystem for Linux 
Информация про датасет в EDA.ipynb  


Download musdb18 and paste it to project folder, folder name should be musdb18
https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems


# Audio Processing Project

This project provides a basic setup for audio processing using Python, with Jupyter Notebook and Streamlit interfaces. It includes Docker support for easy deployment and reproducibility.

## Features

- Audio processing capabilities using torchaudio, librosa, and soundfile
- Interactive web interface using Streamlit
- Jupyter Notebook for detailed analysis
- Docker support for containerized deployment

## Prerequisites

- Docker (for containerized deployment)
- Python 3.9+ (for local development)

## Getting Started

### Using Docker

1. Build the Docker image:
```bash
docker build -t audio-processing .
```

2. Run the container:
```bash
docker run -p 8501:8501 -p 8888:8888 audio-processing
```

3. Access the applications:
   - Streamlit: http://localhost:8501
   - Jupyter Notebook: http://localhost:8888

### Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run Streamlit:
```bash
streamlit run app.py
```

4. Run Jupyter Notebook:
```bash
jupyter notebook
```

## Project Structure

- `app.py`: Streamlit application
- `requirements.txt`: Python dependencies
- `Dockerfile`: Docker configuration

## Libraries Used

- torchaudio: Audio processing with PyTorch
- librosa: Audio and music processing
- soundfile: Audio file I/O
- streamlit: Web application framework
- jupyter: Interactive notebooks 
