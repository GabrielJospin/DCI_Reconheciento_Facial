
# Extensão de arquivo
FORMAT = ".jpg"

# Path para o banco de dados
DATABASE_PATH = "../DataBase"

# Path de exit
TEST_PATH = DATABASE_PATH + "/test"
ORIGINAL_TEST_PATH = TEST_PATH + "/original/"
HOG_TEST_PATH = TEST_PATH + "/HOG/"
LBP_TEST_PATH = TEST_PATH + "/LBP/"

# Path de Treinamento
TRAINING_PATH = DATABASE_PATH + "/training"
ORIGINAL_TRAINING_PATH = TRAINING_PATH + "/original/"
HOG_TRAINING_PATH = TRAINING_PATH + "/HOG/"
LBP_TRAINING_PATH = TRAINING_PATH + "/LBP/"

# pair relation

file_path = [TRAINING_PATH + "/pairsDevTrain" + ".txt",
             TRAINING_PATH + "/notPairsDevTrain" + ".txt",
             TEST_PATH + "/pairsDevTest" + ".txt",
             TEST_PATH + "/notPairsDevTest" + ".txt"]