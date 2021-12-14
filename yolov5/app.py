import os
from connection import s3_connection, s3_put_object
from config import AWS_S3_BUCKET_NAME, AWS_S3_BUCKET_REGION
from flask import Flask, send_from_directory
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
from api.ApiHandler import ApiHandler

s3 = s3_connection()

def auto_upload(file_name):
    filename = file_name
    filepath = './' + filename

    ret1 = s3_put_object(s3, AWS_S3_BUCKET_NAME, './dumpedLog.txt', 'dumpedLog.txt')
    ret2 = s3_put_object(s3, AWS_S3_BUCKET_NAME, filepath, filename)
    if ret1 and ret2:
        print("파일 저장 성공")
    else:
        print("파일 저장 실패")

app = Flask(__name__, static_url_path='', static_folder='frontend/build')
CORS(app)
api = Api(app)

@app.route("/", defaults={'path':''})
def serve(path):
    return send_from_directory(app.static_folder, 'index.html')

api.add_resource(ApiHandler, '/')