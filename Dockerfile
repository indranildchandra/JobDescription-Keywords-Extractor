FROM python:3.6.7
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["src/jobs_indicator_web_services.py"]