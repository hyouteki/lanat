import sys
import torch

def translateT5small(sentence):
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
    inputIds = tokenizer(f"translate English to German: {sentence}",
                          return_tensors="pt").input_ids
    outputs = model.generate(inputIds)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def translateFineTunedT5small(sentence):
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    model = torch.load("setup2c/FineTunedT5SmallModel.pt")
    model.to(DEVICE)
    inputIds = tokenizer(f"translate German to English: {sentence}",
                          return_tensors="pt").input_ids
    outputs = model.generate(inputIds.to(DEVICE))
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    assert len(sys.argv) > 2
    srcSentence = sys.argv[1]
    langauge = sys.argv[2]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if language == "en":
        print(f"setup2b: {translateT5small(srcSentence)}")
    if language == "de":
        print(f"setup2c: {translateFineTunedT5small(srcSentence)}")
