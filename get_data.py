import boto3
import os

# fetch credentials from env variables
aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

# setup a AWS S3 client/resource
s3 = boto3.resource(
    's3', 
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    )

# point the resource at the existing bucket
bucket = s3.Bucket('anyoneai-ay22-01')

#print all object names found in the bucket
for file in bucket.objects.all():
    print(file)

# creating data folder
if not os.path.exists('data'):
    os.makedirs('data')

# download the training dataset
with open('data/training_image_set.tgz', 'wb') as data:
    bucket.download_fileobj('training-datasets/car_ims.tgz', data)

# extracting (to execute in terminal: tar zxvf filename)
import tarfile
images = tarfile.open('data/training_image_set.tgz', 'r')
images.extractall()
images.close()
    
# download the dataset labels
with open('data/car_dataset_labels.csv', 'wb') as data:
    bucket.download_fileobj('training-datasets/car_dataset_labels.csv', data)

# upload a file
# with open('sample.png', 'rb') as data:
#     bucket.upload_fileobj(data, 'raf/sample.png')