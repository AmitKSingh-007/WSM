"""A compact, dependency-free implementation of the original Porter stemmer."""


class PorterStemmer:
    """Reduce English words to stems using Porter's 1980 algorithm."""

    def _consonant(self, word: str, index: int) -> bool:
        letter = word[index]
        if letter in "aeiou":
            return False
        if letter != "y":
            return True
        return index == 0 or not self._consonant(word, index - 1)

    def _measure(self, word: str) -> int:
        """Return the number of vowel-consonant sequences (the Porter m value)."""
        count, previous_was_vowel = 0, False
        for index in range(len(word)):
            is_vowel = not self._consonant(word, index)
            if previous_was_vowel and not is_vowel:
                count += 1
            previous_was_vowel = is_vowel
        return count

    def _has_vowel(self, word: str) -> bool:
        return any(not self._consonant(word, index) for index in range(len(word)))

    def _double_consonant(self, word: str) -> bool:
        return len(word) >= 2 and word[-1] == word[-2] and self._consonant(word, len(word) - 1)

    def _cvc(self, word: str) -> bool:
        if len(word) < 3:
            return False
        end = len(word) - 1
        return (
            self._consonant(word, end)
            and not self._consonant(word, end - 1)
            and self._consonant(word, end - 2)
            and word[end] not in "wxy"
        )

    def _apply(self, word: str, replacements: dict[str, str], minimum_measure: int) -> str:
        for suffix in sorted(replacements, key=len, reverse=True):
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if self._measure(stem) > minimum_measure:
                    return stem + replacements[suffix]
                return word
        return word

    def stem(self, word: str) -> str:
        word = word.lower()
        if len(word) <= 2 or not word.isalpha():
            return word

        # Step 1a
        if word.endswith("sses"):
            word = word[:-2]
        elif word.endswith("ies"):
            word = word[:-2]
        elif word.endswith("ss"):
            pass
        elif word.endswith("s"):
            word = word[:-1]

        # Step 1b
        if word.endswith("eed"):
            stem = word[:-3]
            if self._measure(stem) > 0:
                word = stem + "ee"
        elif word.endswith("ed") and self._has_vowel(word[:-2]):
            word = word[:-2]
            word = self._step_1b_cleanup(word)
        elif word.endswith("ing") and self._has_vowel(word[:-3]):
            word = word[:-3]
            word = self._step_1b_cleanup(word)

        # Step 1c
        if word.endswith("y") and self._has_vowel(word[:-1]):
            word = word[:-1] + "i"

        # Step 2
        word = self._apply(word, {
            "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance",
            "izer": "ize", "abli": "able", "alli": "al", "entli": "ent",
            "eli": "e", "ousli": "ous", "ization": "ize", "ation": "ate",
            "ator": "ate", "alism": "al", "iveness": "ive", "fulness": "ful",
            "ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble",
        }, 0)
        # Step 3
        word = self._apply(word, {
            "icate": "ic", "ative": "", "alize": "al", "iciti": "ic",
            "ical": "ic", "ful": "", "ness": "",
        }, 0)
        # Step 4
        for suffix in ("al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"):
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if suffix != "ion" or (stem and stem[-1] in "st"):
                    if self._measure(stem) > 1:
                        word = stem
                break
        # Step 5a and 5b
        if word.endswith("e"):
            stem = word[:-1]
            measure = self._measure(stem)
            if measure > 1 or (measure == 1 and not self._cvc(stem)):
                word = stem
        if word.endswith("ll") and self._measure(word[:-1]) > 1:
            word = word[:-1]
        return word

    def _step_1b_cleanup(self, word: str) -> str:
        if word.endswith(("at", "bl", "iz")):
            return word + "e"
        if self._double_consonant(word) and word[-1] not in "lsz":
            return word[:-1]
        if self._measure(word) == 1 and self._cvc(word):
            return word + "e"
        return word


_STEMMER = PorterStemmer()


def stem(word: str) -> str:
    """Convenience function used by the preprocessing pipeline."""
    return _STEMMER.stem(word)
