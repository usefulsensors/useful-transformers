from skbuild import setup
setup(
    name='useful-transformers',
    version='0.1',
    description='Efficient Transformer models inference',
    author='Useful Sensors Inc.',
    license='GPLv3',
    package_dir={'useful_transformers': 'examples/whisper'},
    packages=['useful_transformers'],
    package_data={
        'useful_transformers': ['assets/*', 'weights/*'],
    },
    cmake_install_dir='examples/whisper',
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "tiktoken",
        "tqdm",
    ],
)
