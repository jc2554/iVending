"""
Billing: autgenerating html email base on the template and the given customer
info and shopping cart info

MIT License

Copyright (c) 2019 JinJie Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


import smtplib
import ssl
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders


"""
read the email receipt template
input:
    None
output:
    template        'list'      list of each part of the email html template
"""
def read_receipt_template():
    filenames = ['email_template/email_template_part_1.html',
                 'email_template/email_template_part_2.html',
                 'email_template/email_template_part_3.html',
                 'email_template/email_template_before_item.html',
                 'email_template/email_template_mid_item.html',
                 'email_template/email_template_after_item.html',
                 'email_template/email_template_before_total.html',
                 'email_template/email_template_after_total.html']
    template = [""]*8
    for i,filename in enumerate(filenames):
        with open(filename, mode='r') as file:
            template[i] = file.read() # read all lines at once
    return template


"""
generate the receipt email html text base on the template and give info
input:
    name            'string'    name of receiver
    items           'list'      list of items purchased(quantity, item name, unit price)
output:
    email_text      'string'    full body text in html format with transaction info fill into the template
"""
def generate_receipt(name, items, email_template):
    #global email_template
    email_text = """{} {} {} {}{}""".format(email_template[0], str(time.strftime("%c")), email_template[1], name, email_template[2])
    total = 0.0
    for quantity, item_name, price in items:
        email_text += '{}{:d} x {} {}{:0.2f} {}'.format(email_template[3], quantity, item_name, email_template[4], quantity*price, email_template[5])
        total += quantity*price
    email_text += '{}{:.2f} {}'.format(email_template[6], total, email_template[7])
    return email_text


"""
sent transaction receipt email to the customer
input:
    text            'string'    email body text to sent (html format)
    receiver        'dic'       information of receiver(name and email)
output:
    None
"""
def email_receipt(receiver, cart, price_dic, email_template):
    # email parameters
    receiver_email = receiver['email']
    sender_email = "ivending_bot@yahoo.com" 
    password = ""

    # Create message container - the correct MIME type is multipart/alternative.
    msg = MIMEMultipart()#'alternative')
    msg['Subject'] = "iVending Receipt #"+str(int(time.time()))
    msg['From'] = sender_email
    msg['To'] = receiver_email

    msg.preamble = 'This is a multi-part message in MIME format.'
    items = []
    # parse the item count from the cart that has nonnegative count
    for key, val in cart.items():
        if val>0:
            items.append([val, key, price_dic[key]])
    print("billing cart: ", cart)
    print("billing item: ", items)
     #if cart is empty, no receipt to be generated
    if len(items)==0:
        return

    # Create the HTML message
    html = generate_receipt(receiver['name'], items, email_template)

    # Record the MIME types of both parts - text/plain and text/html.
   # part1 = MIMEText(text, 'plain')
    part1 = MIMEText(html, 'html')#, 'us-ascii')

    # Attach HTML message into message container.
    msg.attach(part1)

    # This example assumes the image is in the current directory
    with open('email_template/images/ecommerce-template_order-confirmed-icon.jpg', 'rb') as fp:
        msgImage = MIMEImage(fp.read())

    # Define the image's ID as referenced above
    msgImage.add_header('Content-ID', '<icon>')
    msg.attach(msgImage)

    # Create secure connection with server and send email
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.mail.yahoo.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            print ('Receipt email has sent.')
    except Exception as e:
        print ('[Error] Email sending failed: ',e)