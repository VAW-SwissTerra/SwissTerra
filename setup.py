#!/usr/bin/env python3

from setuptools import setup


setup(name="terra", version="0.0.1", packages=["terra"], entry_points={"console_scripts": ["terra = terra.cli:main"]})
