from django.db import models

# Create your models here.

class Pictures(models.Model):
    pic = models.ImageField(upload_to='booktest/')
    def __str__(self):
        return self.pic