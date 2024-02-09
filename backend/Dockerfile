# Use an official Python runtime as a parent image
FROM python:3.9-slim
WORKDIR /app
COPY . /app/
# COPY app.py /app/
RUN pip install --no-cache-dir -r requirements.txt
# Make port 5000 available to the world outside this container
EXPOSE 5000
CMD ["sh", "-c", "python app.py"]
