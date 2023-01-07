import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='toolbox',
    version='0.0.7',
    author='Alex Valle | Gabriele Berruti',
    author_email='alexvalle75@gmail.com',
    description='Useful package for analyze, manipulate and plot time series',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/RayIci/tstools',
    project_urls = {
    },

    license='MIT',
    packages=['tstools'],
    install_requires=['pandas', 'statsmodels', 'matplotlib'],
)