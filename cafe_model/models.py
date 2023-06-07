from django.db import models
from django.utils import timezone

class CafeResult(models.Model):
    result = models.CharField(max_length = 255)
    timestamp = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.result