from setuptools import setup, find_packages

classes = '''
Intended Audience :: Developers
Intended Audience :: Science/Research
Natural Language :: English
Operating System :: OS Independent
Programming Language :: Python
Programming Language :: Python :: 3
'''
classifiers = [x.strip() for x in classes.splitlines() if x]

setup(
    name="fmmpy",
    version="0.0.1",
    author="Harshvardhan Baldwa",
    author_email="harshbaldwa@gmail.com",
    description="Parallel Fast Multipole Method in Python",
    url="https://github.com/harshbaldwa/fmmpy",
    packages=find_packages(),
    package_data={
        'data': ['fmmpy/data/']
    },
    classifiers=classifiers,
)