from setuptools import find_packages, setup


with open('README.md', encoding='utf-8') as file:
    long_description = file.read()


setup(
    name='pyfoal',
    description='Python forced aligner',
    version='1.0.0',
    author='Max Morrison',
    author_email='maxrmorrison@gmail.com',
    url='https://github.com/maxrmorrison/pyfoal',
    install_requires=[
        'g2p_en',
        'matplotlib',
        'numpy',
        'phonemizer',
        'pypar',
        'requests',
        'scikit-learn',
        'scipy',
        'soundfile',
        'tensorboard',
        'torch<2.0.0',
        'torchaudio<2.0.0',
        'tqdm',
        'yapecs'],
    packages=find_packages(),
    package_data={'pyfoal': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[
        'align',
        'alignment',
        'attention',
        'duration',
        'phoneme',
        'speech',
        'word'],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
