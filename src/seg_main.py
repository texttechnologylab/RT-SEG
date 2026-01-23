import time

from rt_segmentation import (RTLLMBased,
                             RTRuleBased,
                             RTNewLine,
                             bp,
                             sdb_login,
                             load_prompt,
                             load_example_trace)


def test():
    s = time.time()
    res = RTLLMBased.segment_with_sentence_chunks(trace=load_example_trace("trc1"),
                                                  chunk_size=20,
                                                  prompt="",
                                                  system_prompt=load_prompt("system_prompt_sentbased"),
                                                  model_name="Qwen/Qwen2.5-7B-Instruct"
                                                  # model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"

                                                  )
    e = time.time()
    print(res)
    print(e - s)
    for idx, r in enumerate(res):
        # print(trc[r[0]:r[1]])
        # print("-"*10)
        pass



if __name__ == "__main__":
    RTLLMBased.segment()

