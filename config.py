import transformers

################    Train Config    ################
WARM_UP  = 0.00   # Warm up LR 
NON_TOXIX_NUM = 100000
LR = 1e-5

TEST_MODE = False
TRAIN_VAL_COMBINE = True
TRAIN_WITH_2018   = True
TRAIN_FLOAT_SET2  = False
TOXIC_THRESHOLD   = 0.95  # threshold FP targets. (0.95 if TRAIN_FLOAT_SET2=False ; 0.40 if True)
LOSS_WEIGHT       = False

TRAIN_WITH_ALEX   = False
FOCAL_LOSS        = False

# Save Path
SAVE_NAME = "./checkpoints/MaxLen300"

MAX_LEN = 300
ACCUMULATION_STEP = 0

TRAIN_BATCH_SIZE = 200  # GPU10: 400
VALID_BATCH_SIZE = 200 
TEST_BATCH_SIZE  = 1500
EPOCHS = 4


# TRAIN_BATCH_SIZE = 4  # Roberta-base
# VALID_BATCH_SIZE = 4  # 
# TEST_BATCH_SIZE  = 16

TRAIN_WORKERS = 64
VALID_WORKERS = 64
TEST_WORKERS  = 64
PARALLEL = True  # multi-gpu


################    PATH    ################
TRAIN_DATA1 = "../data/jigsaw-toxic-comment-train.csv"
TRAIN_DATA2 = "../data/jigsaw-unintended-bias-train.csv"
VALID_DATA  = "../data/validation.csv"
TEST_DATA   = "../data/test.csv"
SAMPLE_SUB  = "../data/sample_submission.csv"
# 2020 external data
TRAIN_ALEX    = "../data/external_data/train12_trans_alex.csv"
VALID_CAMARON = "../data/external_data/valid_en_camaron.csv"
VALID_YURY    = "../data/external_data/valid_en_yury.csv" 
VALID_SHIROK  = "../data/external_data/valid_en_shirok.csv"
TEST_CAMARON  = "../data/external_data/test_en_camaron.csv" 
TEST_YURY     = "../data/external_data/test_en_yury.csv" 
# 2018 external data
TEST_ZAFAR     = "../data/external_data/18_test_preprocessed_zafar.csv"
TRAIN_ZAFAR    = "../data/external_data/18_train_preprocessed_zafar.csv"
TRAIN_ES_PAVEL = "../data/external_data/18_train_es_pavel.csv"
TRAIN_DE_PAVEL = "../data/external_data/18_train_de_pavel.csv"
TRAIN_FR_PAVEL = "../data/external_data/18_train_fr_pavel.csv"

# ################    PATH    ################
# TRAIN_DATA1 = "../data/jigsaw-toxic-comment-train_clean.csv"
# TRAIN_DATA2 = "../data/jigsaw-unintended-bias-train_clean.csv"
# VALID_DATA  = "../data/validation_clean.csv"
# TEST_DATA   = "../data/test_clean.csv"
# SAMPLE_SUB  = "../data/sample_submission_clean.csv"
# # 2020 external data
# TRAIN_ALEX    = "../data/external_data/train12_trans_alex.csv"
# VALID_CAMARON = "../data/external_data/valid_en_camaron_clean.csv"
# VALID_YURY    = "../data/external_data/valid_en_yury_clean.csv" 
# VALID_SHIROK  = "../data/external_data/valid_en_shirok_clean.csv"
# TEST_CAMARON  = "../data/external_data/test_en_camaron_clean.csv" 
# TEST_YURY     = "../data/external_data/test_en_yury_clean.csv" 
# # 2018 external data
# TEST_ZAFAR     = "../data/external_data/18_test_preprocessed_zafar_clean.csv"
# TRAIN_ZAFAR    = "../data/external_data/18_train_preprocessed_zafar_clean.csv"
# TRAIN_ES_PAVEL = "../data/external_data/18_train_es_pavel_clean.csv"
# TRAIN_DE_PAVEL = "../data/external_data/18_train_de_pavel_clean.csv"
# TRAIN_FR_PAVEL = "../data/external_data/18_train_fr_pavel_clean.csv"


################    BASE MODEL    ################

BERT_PATH = "../pretrained_models/bert-base-multilingual-uncased/"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)


# ROBERTA_PATH = "../pretrained_models/xlm-roberta-base/"
# TOKENIZER = transformers.XLMRobertaTokenizer.from_pretrained(ROBERTA_PATH)


# XLA device
XLA_TRAIN = False
XLA_CORES = 8
