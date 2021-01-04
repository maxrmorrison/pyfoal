from setuptools import setup


with open('README.md') as file:
    long_description = file.read()


setup(
    name='pyfoal',
    description='Python forced aligner',
    version='0.0.1',
    author='Max Morrison',
    author_email='maxrmorrison@gmail.com',
    url='https://github.com/maxrmorrison/pyfoal',
    # TODO - pypar
    install_requires=['g2p_en', 'resampy', 'soundfile'],
    packages=['pyfoal'],
    package_data={'pyfoal': ['assets/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['align', 'duration', 'p2fa', 'phoneme', 'speech'],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
