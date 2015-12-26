# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:21:48 2015

@author: kalial01
"""


from twilio.rest import TwilioRestClient
accountSID = '#######################'
authToken = '############################'
myTwilioNumber = '+########################'
myCellPhone = '+###########################'


def text_myself(message):
    twilioCli = TwilioRestClient(accountSID, authToken) 
    twilioCli.messages.create(body=message, from_=myTwilioNumber, to=myCellPhone) 
    
