from setuptools import setup, find_packages

setup(
    name="tensorflow_gradcam",
    version="0.1.0",
    author="Jeorge Joshi",
    author_email="georgejoshi@example.com",
    description="A package for computing Grad-CAM on TensorFlow models",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wp225/Tensorflow-Gradcam",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow",
        "keras",
        "Pillow",
        "pydantic",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
