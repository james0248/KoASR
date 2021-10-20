# nsml: dacon/nia-pytorch:1.0
from distutils.core import setup

setup(name='ladder_networks',
      version='1.0',
      install_requires=[
          'datasets==1.8.0',
          'transformers==4.4.0',
          'librosa',
          'jiwer',
          'hangul_utils',
          'apex',
      ])
