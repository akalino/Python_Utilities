import boto
import boto.s3
import pandas as pd
import sys
import os
from boto.s3.key import Key


def get_aws_keys(_path):
    aws_df = pd.read_csv(_path)
    key_id = aws_df['AWSAccessKeyId'][0]
    secret_key = aws_df['AWSSecretKey'][0]
    return key_id, secret_key


def percent_complete(complete, total):
    sys.stdout.write('.')
    sys.stdout.flush()


def run():
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY = get_aws_keys('~/Documents/Server/aws_keys.csv')
    bucket_name = AWS_ACCESS_KEY_ID.lower() + '-configs'
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID,
                           AWS_SECRET_ACCESS_KEY)

    bucket = conn.create_bucket(bucket_name,
                                location=boto.s3.connection.Location.DEFAULT)

    file_list = ['/home/alex/Documents/Server/telegraf.conf',
                 '/home/alex/Documents/Server/influxdb.conf',
                 '/home/alex/Documents/Server/emr_bootstrap.sh']

    for _file in file_list:
        print('Uploading {file} to S3 bucket {bucket}'.format(file=_file, bucket=bucket_name))
        k = Key(bucket)
        k.key = os.path.basename(_file)
        k.set_contents_from_filename(_file, cb=percent_complete, num_cb=10)

if __name__ == '__main__':
    run()
