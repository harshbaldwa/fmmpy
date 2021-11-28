from setuptools import setup, find_packages

classes = '''
Development Status :: 2 - Pre-Alpha
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Natural Language :: English
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Topic :: Utilities
'''
classifiers = [x.strip() for x in classes.splitlines() if x]

install_requires = ['scipy', 'pyyaml', 'compyle']
tests_require = ['pytest']
docs_require = ['sphinx', 'sphinx-copybutton']

setup(
    name="fmmpy",
    version="0.0.dev1",
    author="Harshvardhan Baldwa",
    author_email="harshbaldwa@gmail.com",
    description="Parallel Fast Multipole Method in Python",
    long_description=open('README.rst').read(),
    url="https://github.com/harshbaldwa/fmmpy",
    packages=find_packages(),
    license="MIT",
    package_data={
        'data': ['fmmpy/data/']
    },
    classifiers=classifiers,
    install_requires=install_requires,
    extras_require={
        "docs": docs_require,
        "tests": tests_require,
    }
)
