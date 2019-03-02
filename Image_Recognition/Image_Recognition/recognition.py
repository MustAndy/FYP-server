from django.shortcuts import render
import PIL
from PIL import Image
from PIL import ImageDraw
import base64
import json

def recognition_post(request):
    content ={}
    if request.POST:
        with open("./static/images/"+str(request.FILES['picture']),"rb") as f:  
            base64_data = base64.b64encode(f.read())
            image = str(base64_data, encoding='utf-8')
            #result = client.detect(image, imageType, options)

        content['Photo'] = "/static/images/"+str(request.FILES['picture'])
    return render(request, "view.html", content)