from setuptools import setup

setup(
    name='pytorch_beam_search',
    author='Kaiwen Wu',
    license='MIT',
    version='0.1.0',
    packages=['beam_search'],
    python_requires='>=3.9',
    install_requires=[
        'numpy<2.0',
        # We need scaled_dot_product_attention which starts at v2.1.
        'torch>=2.1',
        # We need to use `jax.tree.map`.
        'jax[cpu]',
        # We require the KV caching mechanism up to v4.51.3. After that the
        # mechanism becomes IMO a bit opaque.
        'transformers<=4.51.3',
    ],
    extras_require={
        'dev': [
            'pytest',
            'ruff',
        ],
    },
)
