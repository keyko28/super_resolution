import requests

token = '4b03db604b03db604b03db60cd4b7bd5ba44b034b03db602bb761b83b0509562b9b5af7'
version = 5.131
domain = 'trita.plenka'
offset = 100

response = requests.get('https://api.vk.com/method/wall.get',
                        params={
                            'access_token' : token,
                            'v': version,
                            'domain': domain,
                            'count': 100, # per request
                            'offset': offset
                        })

offset += 100 # next 100

data = response.json()['response']['items']
# to create looped request:
    # take 1000 posts
    # get main film names
    # start looping

print(1)

#TODO
"""
1 Take only popular films (first 10)
2 create the mechanism to avoid several films in one post
kodak portra 160 - good
kodak portra 160 and cinstill 800t - bad
3 make a loop
4 save each film  type in its own directory
"""


