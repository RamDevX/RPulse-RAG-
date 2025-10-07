from django.urls import path
from .views import QueryAPIView 

urlpatterns = [
    path('query/', QueryAPIView.as_view(), name='query',),
]