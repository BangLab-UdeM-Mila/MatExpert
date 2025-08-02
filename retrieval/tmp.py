from transformers import BertTokenizer, BertModel, BertTokenizerFast, Trainer, \
    TrainingArguments, BertForSequenceClassification, BertConfig, BertForMaskedLM, \
    T5Model, T5Tokenizer, T5ForConditionalGeneration, T5Config, T5EncoderModel, AutoTokenizer
from IPython import embed

tokenizer = T5Tokenizer.from_pretrained("t5-base")
# tokenizer = AutoTokenizer.from_pretrained('sagawa/ReactionT5-product-prediction')

chemical_element_symbols = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", 
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", 
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", 
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", 
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", 
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", 
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Th", "Pa", "U", 
    "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", 
    "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", 
    "Ts", "Og"
]

tokenizer.add_tokens(chemical_element_symbols)

desc_text = "Na3MnCoNiO6"

tokens = tokenizer(desc_text, truncation=True, max_length=512, padding="max_length")
token_id, token_mask = tokens["input_ids"], tokens["attention_mask"]

word_num = sum(token_mask)
word_list = tokenizer.convert_ids_to_tokens(token_id)

embed()