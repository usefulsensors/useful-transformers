from skbuild import setup
setup(
    name='useful-transformers',
    version='0.1',
    description='Efficient Transformer models inference',
    author='Useful Sensors Inc.',
    license='GPLv3',
    package_dir={'useful_whisper': 'examples/whisper'},
    packages=['useful_whisper'],
    package_data={
        'useful_whisper': ['assets/*', 'weights/*'],
    },
    cmake_install_dir='examples/whisper',
    python_requires=">=3.7",
)
