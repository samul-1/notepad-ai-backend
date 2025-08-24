"""Add interactions field to Document."""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="document",
            name="interactions",
            field=models.JSONField(blank=True, default=list),
        ),
    ]
