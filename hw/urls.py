from django.urls import path
from . import views

urlpatterns = [
	#path('(showing name)', views.(function name in views), name='裡面用的名稱(?'),
	path('', views.index, name = 'home'),
	path('index', views.index, name = 'index'),
	path('init', views.index, name = 'init'),
	# path('search/<str:tf_form>/<str:text>/<str:type>', views.index, name = 'index'),
	path('article/<str:index>', views.article)
] 
