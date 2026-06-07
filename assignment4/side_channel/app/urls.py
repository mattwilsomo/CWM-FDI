from django.urls import path
from login import views

urlpatterns = [
    path("", views.login_view),
    path("login/", views.login_view, name="login"),
]
