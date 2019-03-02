from django.shortcuts import render
from django.shortcuts import HttpResponse
# Create your views here.
# 上传图片
from django.conf import settings
from Image_upload.models import Pictures
from .reModel import predict
import sys

import os
import PIL

import base64
import json
# Create your views here.
def recognize(request):
    global predict
    predictTemp = predict.Test_predict()
    if request.POST:
    # 从请求当中　获取文件对象
        f1 = request.FILES.get('picture')
        #　利用模型类　将图片要存放的路径存到数据库中
        p = Pictures()
        p.pic = "Image_upload/" + f1.name
        p.save()
        # 在之前配好的静态文件目录static/media/booktest 下 新建一个空文件
        # 然后我们循环把上传的图片写入到新建文件当中
        fname = settings.MEDIA_ROOT + "/Image_upload/" + f1.name
        with open(fname,'wb') as pic:
            for c in f1.chunks():
                pic.write(c)
        
        
        pro, result = predictTemp.predict(fname)
        return render(request,'ImageShow.html',{'pic_obj':p,'result':result})
    