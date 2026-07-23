from setuptools import setup, find_packages

setup(
    name="repair_lm",
    version="0.1.0",
    description="Re-Pair grammar-compression language model with "
                "entity-driven multi-hop composition",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "datasets>=2.19.0",
        "huggingface-hub>=0.23.0",
    ],
    entry_points={
        "console_scripts": [
            "repair-pretrain=pretrain:main",
            "repair-finetune=finetune:main",
            "repair-generate=generate:main",
        ],
    },
)
