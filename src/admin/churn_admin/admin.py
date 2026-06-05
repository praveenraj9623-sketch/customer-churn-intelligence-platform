from django.contrib import admin
from src.admin.churn_admin.models import CustomerRecord


@admin.register(CustomerRecord)
class CustomerRecordAdmin(admin.ModelAdmin):
    list_display = (
        "customer_id", "gender", "tenure", "contract",
        "monthly_charges", "churn_probability", "risk_segment", "created_at"
    )
    search_fields = ("customer_id", "gender", "contract", "risk_segment")
    list_filter = ("gender", "contract", "risk_segment")
