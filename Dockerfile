# Use an official Python runtime as a parent image
FROM python:3.8-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install spaCy and English language model
RUN pip install -r requirements.txt

# Install spaCy and English language model
RUN python train.py --config initial_experiment.yaml

RUN mlflow ui 


EXPOSE 8080

# Run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
