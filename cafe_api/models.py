from django.db import models
from django.contrib.auth.models import User

class Cafe(models.Model):
    name = models.CharField(max_length = 100)
    user = models.ForeignKey(User, on_delete = models.CASCADE, blank = True, null = True)

    def __str__(self):
        return self.name