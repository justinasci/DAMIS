from django.conf.urls import patterns, url, include
from django.conf import settings
from django.conf.urls.i18n import i18n_patterns

from django.contrib.staticfiles.urls import staticfiles_urlpatterns

from damis.views import *


urlpatterns = patterns('',
    (r'^api/', include('damis.api.urls')),
    (r'^i18n/', include('django.conf.urls.i18n')),
)

urlpatterns += i18n_patterns('',
    url(r'^$', index_view, name='home'),
    url(r'^login/$', login_view, name='login'),
    url(r'^logout/$', logout_view, name='logout'),
    url(r'^datasets/$', DatasetList.as_view(), name='dataset-list'),
    url(r'^datasets/new/$', DatasetCreate.as_view(), name='dataset-new'),
    url(r'^datasets/(?P<pk>\d*)/$', DatasetDetail.as_view(), name='dataset-detail'),
    url(r'^datasets/(?P<pk>\d*)/edit/$', DatasetUpdate.as_view(), name='dataset-update'),
    url(r'^datasets/(?P<pk>\d*)/delete/$', DatasetDelete.as_view(), name='dataset-delete'),
    url(r'^datasets/licenses/create/$', DatasetLicenseCreate.as_view(), name='license-new'),

    url(r'^algorithms/$', AlgorithmList.as_view(), name='algorithm-list'),
    url(r'^algorithms/new/$', AlgorithmCreate.as_view(), name='algorithm-new'),
    url(r'^algorithms/(?P<pk>\d*)/edit/$', AlgorithmUpdate.as_view(), name='algorithm-update'),
    url(r'^algorithms/(?P<pk>\d*)/delete/$', AlgorithmDelete.as_view(), name='algorithm-delete'),

    url(r'^experiments/$', ExperimentList.as_view(), name='experiment-list'),
    url(r'^experiments/new/$', ExperimentCreate.as_view(), name='experiment-new'),
    url(r'^experiments/(?P<pk>\d*)/confirm/$', ExperimentDetail.as_view(), name='experiment-confirm'),
)

urlpatterns += staticfiles_urlpatterns()
urlpatterns += patterns('',
    (r'^media/(?P<path>.*)$', 'django.views.static.serve',
     {'document_root': settings.MEDIA_ROOT}),
)
