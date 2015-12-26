# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 13:31:37 2015

@author: kalial01
"""

import smtplib 
smtpObj = smtplib.SMTP('smtp.gmail.com', 587) 
smtpObj.ehlo() #ehlo tries to contact the server, 250 means success
smtpObj.starttls() #enables encription, 220 means ready 

smtpObj.login('kalinowski.aj@gmail.com ', '######')  

sender = 'kalinowski.aj@gmail.com'
receivers = ['kalinowski.aj@gmail.com','Alexander.Kalinowski@Nielsen.com']

message = """From: From Alex <kalinowski.aj@gmail.com>
To: To Alex <kalinowski.aj@gmail.com>
Subject: Model run complete!

This is a test e-mail message.
"""

smtpObj.sendmail(sender, receivers, message)