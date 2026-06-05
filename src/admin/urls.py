from django.contrib import admin
from django.urls import path
from src.admin.churn_admin.views import home

urlpatterns = [
    path("", home, name="home"),
    path("admin/", admin.site.urls),
]
