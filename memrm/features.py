"""Response-text features of the rubric reward model: HEAD, LEN, AXIS and OPT.

The other eight features are LLM scores (scripts/score_features.py). Every feature compares the
memory-conditioned response with the response without memory, and larger values favour the
memory-conditioned response.
"""
import math
import re

HEADING = re.compile(r"^\s*#{1,6}\s", re.M)

# lexical markers of a factual / technical answer
UNIT = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mm|cm|m|km|kg|g|lb|oz|hz|khz|mhz|ghz|w|kw|v|mah|"
                  r"gb|mb|tb|px|dpi|fps|nm|°c|°f|%|inch|inches|\"|')\b", re.I)
MODEL_NUMBER = re.compile(r"\b(?=\w*[A-Za-z])(?=\w*\d)[A-Za-z0-9\-]{4,}\b")
SPEC_WORD = re.compile(r"\b(specification|spec|model|version|compatible|standard|protocol|"
                       r"format|dimension|rating|voltage|capacity|resolution|manual|datasheet)\b", re.I)
# lexical markers of an open-ended advice answer
ADVICE = re.compile(r"\b(consider|try|start by|aim to|recommend|suggestion|you could|you might|"
                    r"it helps to|make sure|focus on|prioriti[sz]e|schedule|routine|habit|"
                    r"plan|goal|balance|tip|strategy|approach)\b", re.I)
IMPERATIVE_BULLET = re.compile(r"^\s*[-*+\d.]+\s*\*{0,2}([A-Z][a-z]+)\b", re.M)
OPTION = re.compile(r"\b(option|alternative|either|instead|depending on|if you|whichever|"
                    r"another way|versus|vs\.?)\b", re.I)


def headings(text):
    return len(HEADING.findall(text))


def word_count(text):
    return len(text.split())


def answer_profile(text):
    """Marker densities per 100 words."""
    n = max(len(re.findall(r"[a-zA-Z']+", text)), 1)
    return {
        "unit": 100.0 * len(UNIT.findall(text)) / n,
        "model_number": 100.0 * len(set(MODEL_NUMBER.findall(text))) / n,
        "spec_word": 100.0 * len(SPEC_WORD.findall(text)) / n,
        "advice": 100.0 * len(ADVICE.findall(text)) / n,
        "imperative": 100.0 * len(IMPERATIVE_BULLET.findall(text)) / n,
        "option": 100.0 * len(OPTION.findall(text)) / n,
    }


def text_features(memory_text, no_memory_text):
    m, w = answer_profile(memory_text), answer_profile(no_memory_text)
    mean = {k: (m[k] + w[k]) / 2 for k in m}
    factual = mean["unit"] + mean["model_number"] + mean["spec_word"]
    advisory = mean["advice"] + mean["imperative"] + mean["option"]
    return {
        "HEAD": headings(memory_text) - headings(no_memory_text),
        "LEN": math.log((word_count(memory_text) + 1) / (word_count(no_memory_text) + 1)),
        # AXIS and OPT describe the pair (averaged over both responses), not a difference
        "AXIS": (advisory - factual) / (advisory + factual + 1e-6),
        "OPT": mean["option"],
    }
