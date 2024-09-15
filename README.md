# Streamlit Image Processing WebApp

This is a Streamlit-based web application for processing images and videos using custom AI models. Users can upload images or videos, select a model, set processing parameters, and download the processed media.

## Features

- **Upload Media**: Supports uploading multiple images or a single video file.
- **Model Selection**: Choose from Model A, Model B, or Model C.
- **Processing Parameters**:
  - Set visualization color using RGB inputs.
  - Adjust video bitrate for output videos.
- **Progress Indicators**: Real-time progress bars during media processing.
- **Persistent Results**: Processed media and download options remain available until new processing starts.
- **Download Results**: Download processed images as a ZIP file or processed video.

## Setup & Usage

```bash
git clone https://github.com/timoillusion/streamlit-image-processing-webapp.git
cd streamlit-image-processing-webapp
```

Startup the Streamlit web application using the following command:
```bash
docker-compose up --build
```

Access at http://localhost:8501 in your web browser.

Shutdown from terminal in the repo directory:

```bash
docker-compose down
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

- **Timo Leitritz**, 2024