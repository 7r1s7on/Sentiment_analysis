FROM python:3.12.7

#adding MODEL_NAME as build argument
ARG MODEL_NAME="final_model.joblib"

#setting the build arguments as environment variable to access through training.py script
ENV MODEL_NAME=${MODEL_NAME}

#changing the path to the working directory in the container
WORKDIR /app

#copy the code itself
COPY src/training /app/src/training

#copy the data_loader 
COPY src/data_loader.py /app/src

#copy utils.py
COPY ../../utils.py /app

#install any necessary packages listed in requirements.txt
COPY ../../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

#loading data
RUN python3 src/data_loader.py

#trains the model after running instance of this image
CMD ["python3","src/training/train.py"]