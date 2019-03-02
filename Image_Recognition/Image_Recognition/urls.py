"""Image_Recognition URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
#from . import recognition
from Image_upload import views as upload_action
from Digit_Recognition import views as digit_recognition
from cifar10_Recognition import views as cifar10_recognition
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

from django.contrib import staticfiles

urlpatterns = [
    #url(r'^recognition$', recognition.recognition_post),
    url(r'^Image_upload/', upload_action.upload),
    url(r'^Digit_result/', digit_recognition.recognize),
    url(r'^Cifar_result/', cifar10_recognition.recognize),
    path('admin/', admin.site.urls),
]
urlpatterns += staticfiles_urlpatterns()
