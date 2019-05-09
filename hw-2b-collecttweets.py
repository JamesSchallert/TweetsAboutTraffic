## James Schallert
## Data Mining
## HW-2B




import twitter, json

##function to print tweets
def print_info(tweet):
    print('***************************')
    print('Tweet ID: ', tweet['id'])
    print('Post Time: ', tweet['created_at'])
    print('User Name: ', tweet['user']['screen_name'])
    try:
        print('Tweet Text: ', tweet['text'])
    except:
        pass



## Crawls tweets with a starting and ending ID that were relevant at the time of this code's conception, but would need updating.  Queries by keywords and geocode, otherwise.  Saves queried tweets in a text file as json objects.
def rest_query1(startingID,stoppingID,myApi):
    raw_tweets = myApi.GetSearch(term= 'traffic OR accident',geocode="42.6526,-73.7562,20mi",count=100,since_id=startingID, max_id=stoppingID)
    for raw_tweet in raw_tweets:
        tweet = json.loads(str(raw_tweet))
        f = open('query1Tweets.txt', 'a+')
        f.write(json.dumps(tweet) + '\n')
        f.close()
        print_info(tweet)
    print(len(raw_tweets))
    
    
    
    
##Same as rest_query1, except it only searches by geocode and not by keyword in order to take a random sample.    
def rest_query2(startingID,stoppingID,myApi):
    raw_tweets = myApi.GetSearch(geocode="42.6526,-73.7562,20mi",count = 100,since_id=startingID, max_id=stoppingID)
    for raw_tweet in raw_tweets:
        tweet = json.loads(str(raw_tweet))
        f = open('query2Tweets.txt', 'a+')
        f.write(json.dumps(tweet) + '\n')
        f.close()
        print_info(tweet)
    print(len(raw_tweets))  




def main():
##Hopefully I never need to use these again, but please don't use them without my permission.
    apiKey = 'H3mA8H68aVttxW8gmaOZKxg1H'
    apiSecretKey = 'Tzot2URbTzkc6IZ2OFt4M23m0eFZPSfwfF7Y8yELAPugSm0c7t'
    accessToken = '1098684654654275586-FmRUJcli65JrXFJfEq4H3rfeCHQhUo'
    tokenSecret = 'd8Wmd6MDHCjFk0GuGtjmXclnqDLpoqgF0PSR2occnGYKA'
    
    myApi = twitter.Api(apiKey, apiSecretKey, accessToken, tokenSecret)
    
##ID that were relevant at the time of this code's conception, but would need updating.  incrementId is a rough estimate of 1/90th of the range between startingID and endingID. 
    startingID = 1096283601832996864
    endingID = 1099869298267435011
    incrementID = 50000000000000
    tempEndingID = startingID+incrementID+1

## Queries the 1/90th of the range of the starting and ending tweet IDs.
    for x in range(1,90):
        print('\nquery1:')
        rest_query1(startingID,tempEndingID,myApi)
        print('\nquery2:')
        rest_query2(startingID,tempEndingID,myApi)
        startingID=tempEndingID+1
        tempEndingID = tempEndingID+incrementID+1

if __name__ == '__main__':
    main()