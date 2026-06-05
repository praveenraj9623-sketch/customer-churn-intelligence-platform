#!/usr/bin/env python
"""Django command-line utility for administrative tasks."""
import os
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "src.admin.settings")
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
