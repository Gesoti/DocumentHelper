"""Embeddings module.

This module is responsible for adding document embeddings in our local VectorStore.
"""

import typing  # pylint: disable=unused-import
import argparse


def main() -> None:
    """Main function is used as entrypoint for `poetry run embeddings` command."""
    # TODO: remove hardcoded url
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"

    # Create argparser and get url if given
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=url)
    args = parser.parse_args()
    print(args)


if __name__ == "__main__":
    main()
