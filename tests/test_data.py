from dft.data import build_messages, tokenize_example


class FakeTokenizer:
    """确定性假 tokenizer：role 首字母 + content 每字符 ord；generation prompt = 999。"""
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        ids = []
        for m in messages:
            ids.append(ord(m["role"][0]))
            ids += [ord(c) for c in m["content"]]
        if add_generation_prompt:
            ids.append(999)
        return ids


def test_build_messages_without_input():
    ex = {"instruction": "你好", "input": "", "output": "您好"}
    msgs = build_messages(ex)
    assert msgs[0] == {"role": "user", "content": "你好"}
    assert msgs[1] == {"role": "assistant", "content": "您好"}


def test_build_messages_with_input():
    ex = {"instruction": "翻译", "input": "cat", "output": "猫"}
    assert build_messages(ex)[0]["content"] == "翻译\n\ncat"


def test_tokenize_masks_prompt_labels():
    tok = FakeTokenizer()
    ex = {"instruction": "hi", "input": "", "output": "yo"}
    out = tokenize_example(ex, tok, max_seq_len=1024)
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}], add_generation_prompt=True)
    plen = len(prompt_ids)
    assert out["labels"][:plen] == [-100] * plen
    assert any(x != -100 for x in out["labels"][plen:])
    assert len(out["input_ids"]) == len(out["labels"]) == len(out["attention_mask"])


def test_tokenize_truncates_to_max_len():
    tok = FakeTokenizer()
    ex = {"instruction": "a" * 50, "input": "", "output": "b" * 50}
    out = tokenize_example(ex, tok, max_seq_len=10)
    assert len(out["input_ids"]) == 10
