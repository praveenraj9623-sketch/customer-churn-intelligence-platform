from django.http import HttpResponse


def home(request):
    return HttpResponse("Customer Churn Django Admin is running. Open /admin/ after creating a superuser.")
