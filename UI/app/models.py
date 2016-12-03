import tweepy

CONSUMER_KEY = "STFpndP02iqX61Q02gSwWSeju"
CONSUMER_SECRET = "bIzoT7jRXxP0jHYK7U2l6BWCZzqz3vOfWrO7WA8EdkOkbY9K7P"
ACCESS_TOKEN = "797174826007834625-hr4WaBsASLneOpHv7vjb2TidFNljkKr"
ACCESS_TOKEN_SECRET = "lTQEGUk5j7gSn7egSoWdSi8i0Fy2PwcjYiWH1MmHIaO0R"


def init_twitter_api():
    '''
    Initializes the tweepy twitter API
    '''
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    return tweepy.API(auth)

def retrieve_model():
    '''
    Retrieves machine learning model from pickle
    '''
    return True
