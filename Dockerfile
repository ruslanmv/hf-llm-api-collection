FROM python:3.11-slim
WORKDIR $HOME/app
COPY . .
RUN pip3 uninstall ffmpeg
RUN pip3 uninstall ffmpeg-python
RUN pip uninstall ffmpeg
RUN pip uninstall ffmpeg-python
RUN pip3 install ffmpeg
RUN pip3 install ffmpeg-python
RUN pip install ffmpeg
RUN pip install ffmpeg-python
RUN pip install -r requirements.txt
VOLUME /data
EXPOSE 23333
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app
RUN mkdir -p $HOME/app/models
RUN chmod 777 $HOME/app/models
ENV MODELS_PATH=$HOME/app/models
RUN mkdir -p $HOME/app/uploads
RUN chmod 777 $HOME/app/uploads

CMD ["python", "-m", "apis.chat_api"]