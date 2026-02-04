import subprocess
import sys
import re

import spacy
from functools import lru_cache

from sympy.categories.baseclasses import Class


class UnitSegmentor:
    patterns = {
            "display_math": r"\$\$.*?\$\$",
            "inline_math": r"\$.*?\$",
            "code_block": r"```.*?```",
            "inline_code": r"`.*?`"
        }

    @staticmethod
    @lru_cache(maxsize=1)
    def load_spacy_model(model_name="en_core_web_sm"):
        """
        Checks if a spaCy model is installed. If not, downloads it.
        Returns the loaded nlp object.
        """
        if not spacy.util.is_package(model_name):
            print(f"Model '{model_name}' not found. Downloading...")
            # Use sys.executable to ensure it installs into the correct environment
            try:
                subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
                print(f"Successfully installed '{model_name}'.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to download model: {e}")
                return None

        return spacy.load(model_name)

    @staticmethod
    def get_math_aware_clauses(text: str):
        # 1. Mask LaTeX/Math: Identify $...$ or $$...$$
        # We use a unique placeholder like MASK_N_
        math_patterns = re.findall(r'(\$\$.*?\$\$|\$.*?\$)', text)
        masked_text = text
        for i, math in enumerate(math_patterns):
            masked_text = masked_text.replace(math, f"MATH_TOKEN_{i}", 1)

        # 2. Standard Clause Segmentation on Masked Text
        doc = UnitSegmentor.load_spacy_model()(masked_text)
        clause_labels = {'ccomp', 'xcomp', 'advcl', 'relcl', 'acl', 'csubj', 'conj'}
        all_clauses = []

        for sent in doc.sents:
            sent_start, sent_end = sent.start_char, sent.end_char
            split_indices = []

            for token in sent:
                if token.dep_ in clause_labels and token.pos_ in {'VERB', 'AUX'}:
                    subtree = sorted(list(token.subtree), key=lambda x: x.i)
                    if subtree:
                        start_idx = subtree[0].idx
                        if sent_start < start_idx < sent_end:
                            split_indices.append(start_idx)

            boundaries = sorted(list(set([sent_start] + split_indices + [sent_end])))

            for i in range(len(boundaries) - 1):
                s, e = boundaries[i], boundaries[i + 1]
                segment = masked_text[s:e]

                # 3. Unmask: Swap placeholders back for original math
                for j, math in enumerate(math_patterns):
                    segment = segment.replace(f"MATH_TOKEN_{j}", math)

                if segment.strip():
                    all_clauses.append((s, e))

        return all_clauses

    @staticmethod
    def get_math_aware_sents(text: str):
        # 1. Identify all protected spans (Math/Code)
        combined_pattern = f"({'|'.join(UnitSegmentor.patterns.values())})"
        matches = list(re.finditer(combined_pattern, text, flags=re.DOTALL))

        # 2. Create a masked version of the text
        masked_text = text
        placeholders = {}

        # We replace from back to front to keep indices valid during string manipulation
        for i, match in enumerate(reversed(matches)):
            placeholder = f"BLOCK_ID_{len(matches) - 1 - i}_"
            start, end = match.span()
            placeholders[placeholder] = match.group()
            masked_text = masked_text[:start] + placeholder + masked_text[end:]

        # 3. Process with spaCy
        doc = UnitSegmentor.load_spacy_model()(masked_text)

        segments = []
        for sent in doc.sents:
            sent_text = sent.text

            # 4. Restore the original content (Unmasking)
            for placeholder, original in placeholders.items():
                if placeholder in sent_text:
                    sent_text = sent_text.replace(placeholder, original)

            segments.append((sent.start_char, sent.end_char))

        return segments

if __name__ == "__main__":
    # Test
    t = """Okay, let's try to figure out this problem. The question is asking for the sum of all integer bases b > 9 where 17_b is a divisor of 97_b. Hmm, first I need to understand what these numbers mean in different bases.\n\nSo, in base b, the number 17_b would be equal to 1*b^1 + 7*b^0, which simplifies to b + 7. Similarly, 97_b is 9*b^1 + 7*b^0, so that's 9b + 7. The problem states that 17_b divides 97_b, meaning that (9b + 7) divided by (b + 7) should be an integer. So, I need to find all bases b > 9 where (9b + 7)/(b + 7) is an integer.\n\nLet me write that division as a fraction: (9b + 7)/(b + 7). Maybe I can simplify this expression. Let's try polynomial division or see if I can rewrite it. Let me see:\n\nIf I divide 9b + 7 by b + 7, maybe I can do some algebra. Let's see:\n\nLet me consider 9b + 7 divided by b + 7. Let's write it as 9*(b + 7) - 63 + 7 divided by (b + 7). Wait, that might not be helpful. Alternatively, maybe express the numerator in terms of the denominator.\n\nLet me think. Let me write 9b + 7 as 9*(b + 7) - 63 + 7. Wait, 9*(b + 7) is 9b + 63. So 9b +7 = 9*(b +7) - 56. Therefore, (9b +7)/(b +7) = 9 - 56/(b +7). So the expression simplifies to 9 - 56/(b +7). For this to be an integer, 56/(b +7) must be an integer. Therefore, (b +7) must be a divisor of 56.\n\nSo, since b > 9, then b +7 must be a positive divisor of 56, and also since b +7 must be greater than 9 +7 = 16? Wait, no. Wait, b is greater than 9, so b +7 is greater than 16. Therefore, we need to find all positive divisors of 56 that are greater than 16, and then subtract 7 from them to get b.\n\nWait, but let me check. Let me list all divisors of 56. The divisors of 56 are 1, 2, 4, 7, 8, 14, 28, 56. But since b +7 must be greater than 16 (since b >9), then the possible divisors are 28, 56. Because 14 is 14, which would make b +7=14 → b=7, but b must be greater than 9. So 14 is too small. Similarly, 8 and 7 are too small. So the only divisors of 56 greater than 16 are 28 and 56. Therefore, b +7 =28 or 56. Therefore, b=21 or b=56-7=49. Wait, 56-7 is 49. So the possible bases are 21 and 49. Therefore, the sum is 21 +49=70.\n\nWait, but let me check if that's correct. Let me verify each case.\n\nFirst, for b=21: 17_b is 1*21 +7=28. 97_b is 9*21 +7=198. 198 divided by 28 is 7. So 198/28=7. So yes, that works.\n\nFor b=49: 17_b is 1*49 +7=56. 97_b is 9*49 +7=441 +7=448. 448 divided by 56 is 8. So that's also an integer. So both cases work.\n\nAre there any other divisors? Let me check again. The divisors of 56 are 1,2,4,7,8,14,28,56. So the ones greater than 16 are 28 and 56. So yes, only those two. Therefore, the answer is 21 +49=70. So the sum is 70.\n\nBut wait, let me make sure there are no other divisors. For example, if b +7 is a divisor of 56, but maybe negative divisors? But since b>9, b+7>16, so divisors have to be positive. So no negative divisors. So yes, only 28 and 56. So the answer is 21 +49=70.\n\nTherefore, the final answer is 70. Let me check if that's correct. Yes, seems right."""
    segments = UnitSegmentor.get_math_aware_sents(t)
    # segments = UnitSegmentor.get_math_aware_clauses(t)
    print(segments)
    """for idx, c in enumerate(segments):
        print(f"{idx} :: {c[0], c[1]} :: ", t[c[0]:c[1]])"""