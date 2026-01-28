from typing import List, Dict, Tuple, Literal
from surrealdb import Surreal, RecordID
from typing import List
import re
from nltk.tokenize import PunktSentenceTokenizer

from .seg_utils import bp, sdb_login, load_prompt, load_example_trace



class RTNewLine:
    @staticmethod
    def paragraph_ranges_regex(text: str) -> list[tuple[int, int]]:
        # Find positions after each \n\n or at start
        positions = [m.end() for m in re.finditer(r'\n\n|\A', text)]
        # Pair consecutive positions
        return list(zip(positions, positions[1:] + [len(text)]))

    @staticmethod
    def new_line_split():
        login_data = sdb_login()
        with Surreal(login_data["url"]) as db:
            db.signin({"username": login_data["user"], "password": login_data["pwd"]})
            db.use(login_data["ns"], login_data["db"])
            db.query("REMOVE TABLE newline_split;")
            db.query("DEFINE TABLE newline_split SCHEMALESS;")
            db.query("DEFINE INDEX idx_newline_split_id ON newline_split FIELDS id;")

            db.query("REMOVE TABLE has_newline_split;")
            db.query("DEFINE TABLE has_newline_split SCHEMALESS TYPE RELATION IN rtrace OUT newline_split;")
            db.query("DEFINE INDEX idx_rt_id ON has_newline_split FIELDS id;")
            db.query("DEFINE INDEX idx_rt_in ON has_newline_split FIELDS in;")
            db.query("DEFINE INDEX idx_rt_out ON has_newline_split FIELDS out;")

            results = db.query("SELECT * from rtrace")

            for res in results:
                rt = res.get("rt")
                offsets = RTNewLine.paragraph_ranges_regex(rt)

                split_id = RecordID("newline_split", res.get("id").id)
                db.upsert(split_id, {"split": offsets})
                db.insert_relation("has_newline_split", {"in": res.get("id"), "out": split_id})