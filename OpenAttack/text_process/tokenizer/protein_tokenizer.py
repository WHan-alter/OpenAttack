from .base import Tokenizer
from ...data_manager import DataManager
from ...tags import *

class ProteinTokenizer(Tokenizer):
    """
    Tokenizer based on ESM alphabet

    Language: amino acids
    """
    TAGS = { TAG_Protein }

    def __init__(self) -> None:
        super().__init__()
