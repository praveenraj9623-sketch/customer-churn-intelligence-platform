from django.db import models


class CustomerRecord(models.Model):
    customer_id = models.CharField(max_length=50, unique=True)
    gender = models.CharField(max_length=20, blank=True)
    tenure = models.IntegerField(default=0)
    contract = models.CharField(max_length=50, blank=True)
    monthly_charges = models.FloatField(default=0.0)
    total_charges = models.FloatField(default=0.0)
    churn_probability = models.FloatField(null=True, blank=True)
    risk_segment = models.CharField(max_length=20, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.customer_id} - {self.risk_segment or 'Not scored'}"
