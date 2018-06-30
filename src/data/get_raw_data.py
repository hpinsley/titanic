import os
from dotenv import load_dotenv, find_dotenv
import logging
import requests
from requests import session

def extract_data(url, file_path):
    """
    Extract data from the kaggle website
    """
    login_url = "https://www.kaggle.com/account/login"

    with session() as c:
        # We need to first get the login page and scrape off the anti-forgery token
        html_page = c.get(login_url).text
        index = html_page.index('antiForgeryToken')
        start = index+19
        end = html_page.index('isAnonymous') - 12
        AFToken = html_page[start:end]
        #print(AFToken)

        payload = {
            'action': 'login',
            'username': os.environ.get("KAGGLE_USERNAME"),
            'password': os.environ.get("KAGGLE_PASSWORD"),
            '__RequestVerificationToken': AFToken
        }

        login_result = c.post(login_url, data=payload)
        #print("Status code is {}".format(login_result.status_code))
        if (login_result.status_code != 200):
            raise Exception("Login result is {}".format(login_result.status_code))
        
        with open(file_path, "w") as f:
            response = c.get(url, stream=True)
            for block in response.iter_content(chunk_size=10000, decode_unicode=True):
                f.write(block)
                        
def main(project_path):
    logger = logging.getLogger(__name__)
    logger.info('Getting raw data')
    
    train_url = "https://www.kaggle.com/c/titanic/download/train.csv"
    test_url = "https://www.kaggle.com/c/titanic/download/test.csv"

    raw_data_path = os.path.join(project_path, "data", "raw")
    train_data_path = os.path.join(raw_data_path, "train.csv")
    test_data_path = os.path.join(raw_data_path, "test.csv")

    extract_data(train_url, train_data_path)
    extract_data(test_url, test_data_path)
    
    logger.info("Downloaded test and training data to {}".format(raw_data_path))
    
if (__name__ == '__main__'):
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    print(project_dir)
    
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    # find .env automatically by walking up directories until it's found
    dotenv_path = find_dotenv()
    # load up the entries as environment variables
    load_dotenv(dotenv_path)

    main(project_dir)    
    