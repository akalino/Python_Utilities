# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:21:48 2015

@author: kalial01
"""


from twilio.rest import TwilioRestClient
accountSID = 'ACa9b83f8b99a5365d30748c464ae3220d'
authToken = '7e15e3a96285446ee1a3ac06b53661c4'
myTwilioNumber = '+14155992671'
myCellPhone = '+16094059238'


def text_myself(message):
    twilioCli = TwilioRestClient(accountSID, authToken) 
    twilioCli.messages.create(body=message, from_=myTwilioNumber, to=myCellPhone) 
    
