from setuptools import setup

setup(
    name='dimon-cli',
    version='0.1.0',
    py_modules=['dimon'],
    description='dimensional monitoring cli',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='InfuseAI',
    url='https://github.com/InfuseAI/dimon',
    install_requires=[
        'Click',
        'pandas',
        'python-dotenv',
        'llama-index',
        'llama-index-vector-stores-elasticsearch',
        'llama-index-llms-ollama',
        'llama-index-embeddings-ollama',
        'rich',
    ],
    entry_points={
        'console_scripts': ['dimon = dimon.main:cli']
    },
)