from django.contrib import admin
from . import models #在admin中导入models
# Register your models here.
admin.site.register(models.NLP_SOURCE) #将数据库在admin中注册