from django.shortcuts import render
from django.shortcuts import HttpResponse
# Create your views here.
# 上传图片
from django.conf import settings
from .models import Pictures

import PIL

import base64
import json

# 返回上传图片的页面
def getUpload(request):
    return render(request, "view.html")

#　发来表单　实现上传功能
def upload(request):
    return render(request, "view.html")

    