# --- task structure -------------------------------------------------------- #
INPUT_VARS = ("W", "X", "Y", "Z")     # input slots, in concatenation order
TARGET_VARS = ("WX", "YZ")            # abstract variables we intervene on
NUM_CLASSES = 2                       # O is binary
 
# --- entity table / input encoding ----------------------------------------- #
NUM_ENTITIES = 20                     # size of the discrete entity table
EMBEDDING_DIM = 4                     # dim of each entity vector
INPUT_DIM = len(INPUT_VARS) * EMBEDDING_DIM   # 16: four concatenated entity vectors
 
# --- backbone architecture -------------------------------------------------- #
HIDDEN_DIMS = (16, 16, 16)            # three hidden layers -> 3 * 16 = 48 neuron sites
ACTIVATION = "relu"
 
# --- training data sizes ---------------------------------------------------- #
FACTUAL_TRAIN_SIZE = 30_000
FACTUAL_VAL_SIZE = 4_000
 
# --- intervention pair banks ------------------------------------------------ #
PAIR_BANK_TRAIN_SIZE = 4_000          # per target variable, for fitting OT
PAIR_BANK_TEST_SIZE = 2_000           # per target variable, held-out for scoring
 
# --- artifacts -------------------------------------------------------------- #
CHECKPOINT_PATH = "models/equality_mlp.pt"   # relative to the current working dir