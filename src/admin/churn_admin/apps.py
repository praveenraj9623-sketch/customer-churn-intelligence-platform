from django.apps import AppConfig


class ChurnAdminConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "src.admin.churn_admin"
    verbose_name = "Customer Churn Admin"
