from setuptools import setup
import os


def find_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            relative_path = os.path.relpath(os.path.join(path, filename), directory)
            paths.append(os.path.join(directory, relative_path))
    return paths


setup(
    name='pySLAMM',
    version='6.7.0',
    author='Warren Pinnacle Consulting, Inc.',
    author_email='jclough@warrenpinnacle.com',
    description='Translation of SLAMM 6.7 to Python without GUI. Reads txt project files from SLAMM 6.7 Delphi',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/tba',
    py_modules=[os.path.splitext(f)[0] for f in os.listdir('.') if f.endswith('.py')],
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
    install_requires=[
        'setuptools~=69.5.1',
        'numba~=0.59.1',
        'pandas~=2.2.2',
        'numpy~=1.26.4',
        'dbf~=0.99.9',
        'shapely~=2.0.4',
        'rasterio~=1.3.10',
    ],
    include_package_data=True,
    package_data={
        '': ['*.txt'],
        'Kakahaia': ['*'],
        'docs': ['*'],
    },
    data_files=[
        ('Kakahaia', find_files('Kakahaia')),
        ('docs', find_files('docs')),
    ],
)
