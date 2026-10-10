#!/usr/bin/env python3
"""
Determine most similar words in terms of their word embeddings.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from integerize import Integerizer   # look at integerize.py for more info

# Needed for Python's optional type annotations.
# We've included type annotations and recommend that you do the same, 
# so that mypy (or a similar package) can catch type errors in your code.
from typing import List, Optional

try:
    # PyTorch is your friend. Not using it will make your program so slow.
    # And it's also required for this assignment. ;-)
    # So if you comment this block out instead of dealing with it, you're
    # making your own life worse.
    import torch
    import torch.nn as nn
except ImportError:
    print("\nERROR! You need to install Miniconda, then create and activate the nlp-class environment.  See the INSTRUCTIONS file.\n")
    raise


log = logging.getLogger(Path(__file__).stem)  # The only okay global variable.

# Logging is in general a good practice to monitor the behavior of your code
# while it's running. Compared to calling `print`, it provides two benefits.
# 
# - It prints to standard error (stderr), not standard output (stdout) by
#   default.  So these messages will normally go to your screen, even if
#   you have redirected stdout to a file.  And they will not be seen by
#   the autograder, so the autograder won't be confused by them.
# 
# - You can configure how much logging information is provided, by
#   controlling the logging 'level'. You have a few options, like
#   'debug', 'info', 'warning', and 'error'. By setting a global flag,
#   you can ensure that the information you want - and only that info -
#   is printed. As an example:
#        >>> try:
#        ...     rare_word = "prestidigitation"
#        ...     vocab.get_counts(rare_word)
#        ... except KeyError:
#        ...     log.error(f"Word that broke the program: {rare_word}")
#        ...     log.error(f"Current contents of vocab: {vocab.data}")
#        ...     raise  # Crash the program; can't recover.
#        >>> log.info(f"Size of vocabulary is {len(vocab)}")
#        >>> if len(vocab) == 0:
#        ...     log.warning(f"Empty vocab. This may cause problems.")
#        >>> log.debug(f"The values are {vocab}")
#   If we set the log level to be 'INFO', only the log.info, log.warning,
#   and log.error statements will be printed. You can calibrate exactly how 
#   much info you need, and when. None of these pollute stdout with things 
#   that aren't the real 'output' of your program.
# 
# In `parse_args`, we provided two command-line options to control the logging level.
# The default level is 'INFO'. You can lower it to 'DEBUG' if you pass '--verbose'
# and you can raise it to 'WARNING' if you pass '--quiet'.
#
# More info: https://docs.python.org/3/howto/logging.html#logging-basic-tutorial
# 
# In all the starter code for the NLP course, we've elected to create a separate
# logger for each source code file, stored in a variable named log that
# is globally visible throughout the file.  That way, calls like log.info(...)
# will use the logger for the current source code file and thus their output will 
# helpfully show the filename.  You could configure the current file's logger using
# log.basicConfig(...), whereas logging.basicConfig(...) affects all of the loggers.
# The command-line options affect all of the loggers.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings", type=Path, help="path to word embeddings file")
    parser.add_argument("word", type=str, help="word to look up")
    parser.add_argument("--minus", type=str, default=None)
    parser.add_argument("--plus", type=str, default=None)

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()
    if not args.embeddings.is_file():
        parser.error(f"Embeddings file {args.embeddings} not found")
    if (args.minus is None) != (args.plus is None):  # != is the XOR operation!
        parser.error("Must include both `--plus` and `--minus` or neither")

    return args

class Lexicon:
    """
    Class that manages a lexicon and can compute similarity.

    >>> my_lexicon = Lexicon.from_file(my_file)
    >>> my_lexicon.find_similar_words("bagpipe")
    """

    def __init__(self, vocab: Integerizer[str], embeddings: torch.Tensor) -> None:
        """Load information into coupled word-index mapping and embedding matrix."""
        if embeddings.ndim != 2 or embeddings.shape[0] != len(vocab):
            raise ValueError("Vocabulary and embedding matrix have incompatible sizes")
        self.vocab = vocab
        self.embeddings = embeddings
        self.normalized_embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

    @classmethod
    def from_file(cls, file: Path) -> Lexicon:
        with open(file) as f:
            first_line = next(f) 
            try:
                num_words, dimensions = map(int, first_line.split())
            except ValueError as error:
                raise ValueError(
                    f"first line of must contain vocabulary and dimension sizes"
                ) from error

            words: List[str] = []
            vectors: List[List[float]] = []
            for line in f: 
                fields = line.split()
                if not fields:
                    continue
                word, *values = fields
                if len(values) != dimensions:
                    raise ValueError(
                        f"error {dimensions} values for {word}, got {len(values)}"
                    )
                words.append(word)
                vectors.append([float(value) for value in values])

        if len(words) != num_words:
            raise ValueError(f"Expected {num_words} words, got {len(words)}")
        return cls(Integerizer(words), torch.tensor(vectors, dtype=torch.float32))

    def find_similar_words(
        self, word: str, *, plus: Optional[str] = None, minus: Optional[str] = None
    ) -> List[str]:
        """Find most similar words, in terms of embeddings, to a query."""
        # FINISH THIS FUNCTION

        # The star above forces you to use `plus` and `minus` only
        # as named arguments. This helps avoid mixups or readability
        # problems where you forget which comes first.
        #
        # We've also given `plus` and `minus` the type annotation
        # Optional[str]. This means that the argument may be None, or
        # it may be a string. If you don't provide these, it'll automatically
        # use the default value we provided: None.
        if (minus is None) != (plus is None):  # != is the XOR operation!
            raise TypeError("Must include both of `plus` and `minus` or neither.")

        word_index = self.vocab.index(word)
        if word_index is None:
            raise ValueError(f"error word not found in lexicon: {word}")

        query = self.embeddings[word_index].clone()
        excluded_indices = [word_index]
        if minus is not None and plus is not None:
            minus_index = self.vocab.index(minus)
            plus_index = self.vocab.index(plus)
            if minus_index is None:
                raise ValueError(f"error word not found in lexicon: {minus}")
            if plus_index is None:
                raise ValueError(f"error word not found in lexicon: {plus}")
            query = query - self.embeddings[minus_index] + self.embeddings[plus_index]
            excluded_indices.extend((minus_index, plus_index))

        query = nn.functional.normalize(query, p=2, dim=0)
        similarities = self.normalized_embeddings @ query
        similarities[torch.tensor(excluded_indices)] = -torch.inf
        count = min(10, len(self.vocab) - len(set(excluded_indices)))
        best_indices = torch.topk(similarities, k=count).indices.tolist()
        return list(map(self.vocab.__getitem__, best_indices))


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    lexicon = Lexicon.from_file(args.embeddings)
    similar_words = lexicon.find_similar_words(
        args.word, plus=args.plus, minus=args.minus
    )
    print(" ".join(similar_words))  # print all words on one line, separated by spaces


if __name__ == "__main__":
    main()
