import transformers

################    Train Config    ################
TOXIC_THRESHOLD = 0.95  # used for thresholding {val_in_train, val_traget} in training.
WARM_UP  = 0.00   # Warm up LR 
NON_TOXIX_NUM   = 100000
LR = 1e-5

TEST_MODE = True
TRAIN_VAL_COMBINE = True
TRAIN_WITH_2018 = False
TRAIN_WITH_ALEX = False
FOCAL_LOSS = False

# Save Path
SAVE_NAME = "./checkpoints/reborta_base"

MAX_LEN = 192
TRAIN_BATCH_SIZE = 4  # if on TPU: 128 (16*8cores)
VALID_BATCH_SIZE = 4  # if on TPU: 128 (16*8cores)
TEST_BATCH_SIZE  = 16
EPOCHS = 6

# TRAIN_BATCH_SIZE = 400  # if on TPU: 128 (16*8cores)
# VALID_BATCH_SIZE = 400  # if on TPU: 128 (16*8cores)
# TEST_BATCH_SIZE  = 3200

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


################    BASE MODEL    ################

BERT_PATH = "../pretrained_models/bert-base-multilingual-uncased/"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)


# ROBERTA_PATH = "../pretrained_models/xlm-roberta-base/"
# TOKENIZER = transformers.XLMRobertaTokenizer.from_pretrained(ROBERTA_PATH)



# XLA device
XLA_TRAIN = False
XLA_CORES = 8
