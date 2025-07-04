import logging
import os
from datetime import datetime


# log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# logs_path = os.path.join(os.getcwd() , "logs " , log_file)
# os.makedirs(logs_path , exist_ok=True)

# log_file_path = os.path.join(logs_path , log_file)

# logging.basicConfig(

#     filemode=log_file_path,
#     format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
#     level=logging.INFO
# )

import logging
import os
from datetime import datetime

log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_dir = os.path.join(os.getcwd(), "logs")  # Directory ONLY (no filename)
os.makedirs(logs_dir, exist_ok=True)  # create logs folder

log_file_path = os.path.join(logs_dir, log_file)  # Full path to the log file

logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

if __name__=="__main__":
    logging.info("logging is started")




# if __name__=="__main__":

#     logging.INFO("loging is started")