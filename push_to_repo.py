from transformers import AutoTokenizer, AutoConfig, EncoderDecoderModel, AutoModelForSeq2SeqLM, MBartTokenizerFast

model_repo_name = "combined_tr_berturk32k_cased_summary"
model_name_or_path = "outputs/checkpoint-82325"
config = AutoConfig.from_pretrained(
    model_name_or_path
)
if "mbart" in model_name_or_path:
    tokenizer = MBartTokenizerFast.from_pretrained(
        model_name_or_path,
        src_lang="tr_TR",
        tgt_lang="tr_TR")
else:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        strip_accents=False,
        lowercase=False
    )

if "bert" in model_name_or_path:
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

if "bert" in model_name_or_path:
    model = EncoderDecoderModel.from_pretrained(model_name_or_path)
    # set special tokens
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # sensible parameters for beam search
    model.config.vocab_size = model.config.decoder.vocab_size
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
        config=config,
    )

if "mbart" in model_name_or_path:
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

if model.config.decoder_start_token_id is None:
    raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

model.push_to_hub(model_repo_name,use_temp_dir=True)