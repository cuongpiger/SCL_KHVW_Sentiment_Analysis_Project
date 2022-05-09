from django.urls import path
from .views import (
    home,
    get_predicted_sentiment,
    ajax_predict_comment
)


app_name = 'sentiment'
urlpatterns = [
    path('', home, name='home'),
    path('get-predicted-sentiment/', get_predicted_sentiment, name='get_predicted_sentiment'),
    path('ajax-predict-comment/', ajax_predict_comment, name='ajax_predict_comment'),
]
