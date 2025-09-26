# Step 1: Use the immutable SHA256 digest for 100% reproducibility
FROM nvcr.io/nvidia/tensorflow@sha256:fdc2f7f3f63c47d71dff5646f26d9c922aeeb5d477a8594a5cd24928a9d5e82e

# Step 2: Install any needed system libraries using apt
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    screen \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Copy only the requirements file to leverage Docker's build cache
COPY requirements.txt /app/requirements.txt

# Step 4: Install Python packages using the pinned and hashed requirements file
RUN pip install --no-cache-dir --require-hashes -r /app/requirements.txt

# Step 5: Copy the rest of the application source code
COPY . /app

# Step 6: Add git safe.directory config to bash startup script for the root user
# This will run every time you start an interactive shell (e.g., 'docker run... bash')
RUN echo "git config --global --add safe.directory /app" >> /root/.bashrc

# Step 7: Set the working directory
WORKDIR /app
