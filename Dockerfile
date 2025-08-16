FROM python:3.11

WORKDIR /src

COPY requirement.txt requirement.txt

#install all library in requirement.txt
RUN pip install --no-cache-dir -r requirement.txt

#copy all files from local to container directory
COPY . .

#set port for the container
EXPOSE 8000
EXPOSE 8001

#send command to check python version
CMD ["python", "--version"]