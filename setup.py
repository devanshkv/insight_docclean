from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='insight_docclean',
    version='0.1',
    packages=find_packages(),
    tests_require=['pytest', 'pytest-cov'],
    zip_safe=False,
    url='https://github.com/devanshkv/insight_docclean/',
    license='GNU Lesser General Public License v3.0',
    author='Devansh Agarwal',
    author_email='devansh.kv@gmail.com',
    description='Clean your document images',
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)']
)
